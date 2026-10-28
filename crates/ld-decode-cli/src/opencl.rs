//! OpenCL backend of the stack's virtual source (`--stack-device opencl`).
//!
//! The OpenCL library is loaded at run time (`OpenCL.dll`, shipped with the
//! GPU driver on Windows; `libOpenCL.so.1` elsewhere), so building needs no
//! SDK and a machine without it only loses this option. The kernels evaluate
//! exactly the plan the CPU path evaluates (`stack::Plan`): the same positions
//! (in double precision), the same sinc phase, the same tables, in single
//! precision for the dot products.

use std::ffi::{c_char, c_void, CString};
use std::ptr::{null, null_mut};
use std::sync::Mutex;

use anyhow::{bail, Context as _, Result};
use libloading::Library;

use crate::stack::{sinc_lut, Inner, Plan, HI_TAPS, LOCK_WIN, SINC_PHASES, SINC_TAPS};

type ClInt = i32;
type ClUint = u32;
type Handle = *mut c_void;

const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_COPY_HOST_PTR: u64 = 1 << 5;
const CL_TRUE: u32 = 1;

/// The OpenCL 1.2 entry points used here.
struct Api {
    _lib: Library,
    get_platform_ids: unsafe extern "system" fn(ClUint, *mut Handle, *mut ClUint) -> ClInt,
    get_device_ids: unsafe extern "system" fn(Handle, u64, ClUint, *mut Handle, *mut ClUint) -> ClInt,
    get_device_info: unsafe extern "system" fn(Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_context: unsafe extern "system" fn(*const isize, ClUint, *const Handle, *const c_void, *mut c_void, *mut ClInt) -> Handle,
    create_command_queue: unsafe extern "system" fn(Handle, Handle, u64, *mut ClInt) -> Handle,
    create_program_with_source: unsafe extern "system" fn(Handle, ClUint, *const *const c_char, *const usize, *mut ClInt) -> Handle,
    build_program: unsafe extern "system" fn(Handle, ClUint, *const Handle, *const c_char, *const c_void, *mut c_void) -> ClInt,
    get_program_build_info: unsafe extern "system" fn(Handle, Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_kernel: unsafe extern "system" fn(Handle, *const c_char, *mut ClInt) -> Handle,
    create_buffer: unsafe extern "system" fn(Handle, u64, usize, *mut c_void, *mut ClInt) -> Handle,
    release_mem_object: unsafe extern "system" fn(Handle) -> ClInt,
    set_kernel_arg: unsafe extern "system" fn(Handle, ClUint, usize, *const c_void) -> ClInt,
    enqueue_write_buffer: unsafe extern "system" fn(Handle, Handle, u32, usize, usize, *const c_void, ClUint, *const Handle, *mut Handle) -> ClInt,
    enqueue_read_buffer: unsafe extern "system" fn(Handle, Handle, u32, usize, usize, *mut c_void, ClUint, *const Handle, *mut Handle) -> ClInt,
    enqueue_nd_range_kernel: unsafe extern "system" fn(Handle, Handle, ClUint, *const usize, *const usize, *const usize, ClUint, *const Handle, *mut Handle) -> ClInt,
    finish: unsafe extern "system" fn(Handle) -> ClInt,
}

impl Api {
    fn load() -> Result<Self> {
        let names: &[&str] = if cfg!(windows) {
            &["OpenCL.dll"]
        } else if cfg!(target_os = "macos") {
            &["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
        } else {
            &["libOpenCL.so.1", "libOpenCL.so"]
        };
        let lib = names
            .iter()
            .find_map(|n| unsafe { Library::new(n) }.ok())
            .context("no OpenCL runtime found (it comes with the GPU driver)")?;
        // Copy the entry points out before `lib` moves into the struct.
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .with_context(|| format!("OpenCL runtime lacks {}", $name))?
            };
        }
        let get_platform_ids = sym!("clGetPlatformIDs");
        let get_device_ids = sym!("clGetDeviceIDs");
        let get_device_info = sym!("clGetDeviceInfo");
        let create_context = sym!("clCreateContext");
        let create_command_queue = sym!("clCreateCommandQueue");
        let create_program_with_source = sym!("clCreateProgramWithSource");
        let build_program = sym!("clBuildProgram");
        let get_program_build_info = sym!("clGetProgramBuildInfo");
        let create_kernel = sym!("clCreateKernel");
        let create_buffer = sym!("clCreateBuffer");
        let release_mem_object = sym!("clReleaseMemObject");
        let set_kernel_arg = sym!("clSetKernelArg");
        let enqueue_write_buffer = sym!("clEnqueueWriteBuffer");
        let enqueue_read_buffer = sym!("clEnqueueReadBuffer");
        let enqueue_nd_range_kernel = sym!("clEnqueueNDRangeKernel");
        let finish = sym!("clFinish");
        Ok(Self {
            _lib: lib,
            get_platform_ids,
            get_device_ids,
            get_device_info,
            create_context,
            create_command_queue,
            create_program_with_source,
            build_program,
            get_program_build_info,
            create_kernel,
            create_buffer,
            release_mem_object,
            set_kernel_arg,
            enqueue_write_buffer,
            enqueue_read_buffer,
            enqueue_nd_range_kernel,
            finish,
        })
    }
}

fn check(code: ClInt, what: &str) -> Result<()> {
    if code != 0 {
        bail!("OpenCL {what} failed (error {code})");
    }
    Ok(())
}

const SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

// One input read at its virtual samples: the full signal (sinc) and the
// analytic video band, as stack.rs compute_task does.
__kernel void read_input(
    __global const float* raw,
    __global const uint* seg,
    __global const double* seg_t0,
    __global const double* a0,
    __global const double* slope,
    const double s0,
    const int len,
    const int base,
    __global const float* sinc,
    __global const float2* hil,
    __global float* full,
    __global float2* hi,
    __global uchar* cov)
{
    int j = get_global_id(0);
    if (j >= len) return;
    uint k = seg[j];
    double a = a0[k];
    if (isnan(a)) {
        full[base + j] = 0.0f;
        hi[base + j] = (float2)(0.0f, 0.0f);
        cov[base + j] = 0;
        return;
    }
    double p = a + ((s0 + (double)j) - seg_t0[k]) * slope[k];
    double kf = floor(p);
    int k0 = (int)kf;
    int ph = (int)round((p - kf) * (double)PHASES);
    __global const float* xs = raw + (k0 - (SINC_TAPS / 2 - 1));
    __global const float* hs = sinc + ph * SINC_TAPS;
    float f = 0.0f;
    for (int d = 0; d < SINC_TAPS; d++) f += hs[d] * xs[d];
    __global const float* xh = raw + (k0 - (HI_TAPS / 2 - 1));
    __global const float2* hh = hil + ph * HI_TAPS;
    float2 h = (float2)(0.0f, 0.0f);
    for (int d = 0; d < HI_TAPS; d++) h += hh[d] * xh[d];
    full[base + j] = f;
    hi[base + j] = h;
    cov[base + j] = 1;
}

// Phase-lock every input's video band onto the primary's over LOCK_WIN
// samples and sum with the weights (the rest of each input as read).
__kernel void combine(
    const int ni,
    const int len,
    const int m,
    const int n,
    __global const float* full,
    __global const float2* hi,
    __global const uchar* cov,
    __global const float* weight,
    __global float* out)
{
    int jo = get_global_id(0);
    if (jo >= n) return;
    int j = jo + m;
    float acc = 0.0f, wsum = 0.0f;
    if (cov[j]) { acc += full[j] * weight[0]; wsum += weight[0]; }
    int lo = max(0, j - LOCK_WIN / 2), up = min(len, j + LOCK_WIN / 2);
    for (int i = 1; i < ni; i++) {
        int b = i * len;
        if (!cov[b + j]) continue;
        float2 r = (float2)(0.0f, 0.0f);
        for (int t = lo; t < up; t++) {
            if (cov[t] && cov[b + t]) {
                float2 x = hi[t], y = hi[b + t];
                r += (float2)(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y);
            }
        }
        float a = length(r);
        float2 rot = a > 0.0f ? r / a : (float2)(1.0f, 0.0f);
        float2 d = rot - (float2)(1.0f, 0.0f);
        float2 z = hi[b + j];
        acc += (full[b + j] + (z.x * d.x - z.y * d.y)) * weight[i];
        wsum += weight[i];
    }
    out[jo] = wsum > 0.0f ? acc / wsum : 0.0f;
}
"#;

/// A device buffer reused across reads, grown on demand.
struct Buf {
    mem: Handle,
    cap: usize,
}

struct State {
    queue: Handle,
    context: Handle,
    k_read: Handle,
    k_combine: Handle,
    sinc: Handle,
    hil: Handle,
    weight: Handle,
    seg: Buf,
    seg_t0: Buf,
    raw: Buf,
    a0: Buf,
    slope: Buf,
    full: Buf,
    hi: Buf,
    cov: Buf,
    out: Buf,
}

pub struct StackGpu {
    api: Api,
    ni: usize,
    state: Mutex<State>,
}

// The handles are only used under the mutex; OpenCL objects themselves are
// thread-safe to share.
unsafe impl Send for StackGpu {}
unsafe impl Sync for StackGpu {}

impl StackGpu {
    pub(crate) fn new(inner: &Inner) -> Result<Self> {
        let api = Api::load()?;
        unsafe {
            let mut np = 0;
            check((api.get_platform_ids)(0, null_mut(), &mut np), "clGetPlatformIDs")?;
            let mut platforms = vec![null_mut(); np as usize];
            check((api.get_platform_ids)(np, platforms.as_mut_ptr(), null_mut()), "clGetPlatformIDs")?;
            let mut devices = Vec::new();
            for &p in &platforms {
                let mut nd = 0;
                if (api.get_device_ids)(p, CL_DEVICE_TYPE_GPU, 0, null_mut(), &mut nd) != 0 || nd == 0 {
                    continue;
                }
                let mut d = vec![null_mut(); nd as usize];
                check((api.get_device_ids)(p, CL_DEVICE_TYPE_GPU, nd, d.as_mut_ptr(), null_mut()), "clGetDeviceIDs")?;
                devices.extend(d);
            }
            let pick = std::env::var("LD_OPENCL_DEVICE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
            let Some(&device) = devices.get(pick) else {
                bail!("no OpenCL GPU device (found {})", devices.len());
            };
            let info = |what: u32| -> Result<String> {
                let mut size = 0;
                check((api.get_device_info)(device, what, 0, null_mut(), &mut size), "clGetDeviceInfo")?;
                let mut buf = vec![0u8; size];
                check((api.get_device_info)(device, what, size, buf.as_mut_ptr().cast(), null_mut()), "clGetDeviceInfo")?;
                Ok(String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string())
            };
            let name = info(CL_DEVICE_NAME)?;
            if !info(CL_DEVICE_EXTENSIONS)?.contains("cl_khr_fp64") {
                bail!("OpenCL device {name} has no double precision (cl_khr_fp64), which the positions need");
            }
            tracing::info!("stack: OpenCL device {name}");

            let mut err = 0;
            let context = (api.create_context)(null(), 1, &device, null(), null_mut(), &mut err);
            check(err, "clCreateContext")?;
            let queue = (api.create_command_queue)(context, device, 0, &mut err);
            check(err, "clCreateCommandQueue")?;
            let src = CString::new(SOURCE)?;
            let ptr = src.as_ptr();
            let program = (api.create_program_with_source)(context, 1, &ptr, null(), &mut err);
            check(err, "clCreateProgramWithSource")?;
            let opts = CString::new(format!(
                "-DPHASES={SINC_PHASES} -DSINC_TAPS={SINC_TAPS} -DHI_TAPS={HI_TAPS} -DLOCK_WIN={LOCK_WIN}"
            ))?;
            if (api.build_program)(program, 1, &device, opts.as_ptr(), null(), null_mut()) != 0 {
                let mut size = 0;
                (api.get_program_build_info)(program, device, CL_PROGRAM_BUILD_LOG, 0, null_mut(), &mut size);
                let mut log = vec![0u8; size];
                (api.get_program_build_info)(program, device, CL_PROGRAM_BUILD_LOG, size, log.as_mut_ptr().cast(), null_mut());
                bail!("OpenCL kernel build failed:\n{}", String::from_utf8_lossy(&log));
            }
            let k_read = (api.create_kernel)(program, c"read_input".as_ptr(), &mut err);
            check(err, "clCreateKernel")?;
            let k_combine = (api.create_kernel)(program, c"combine".as_ptr(), &mut err);
            check(err, "clCreateKernel")?;

            let sinc: Vec<f32> = sinc_lut().iter().map(|&v| v as f32).collect();
            let hil: Vec<f32> = inner.hi_lut.iter().flat_map(|c| [c.re as f32, c.im as f32]).collect();
            let wmax = inner.weight.iter().copied().fold(0.0, f64::max);
            let weight: Vec<f32> = inner.weight.iter().map(|&w| (w / wmax) as f32).collect();
            let constant = |data: &[f32]| -> Result<Handle> {
                let mut err = 0;
                let mem = (api.create_buffer)(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    std::mem::size_of_val(data),
                    data.as_ptr() as *mut c_void,
                    &mut err,
                );
                check(err, "clCreateBuffer")?;
                Ok(mem)
            };
            let empty = || Buf { mem: null_mut(), cap: 0 };
            let state = State {
                queue,
                context,
                k_read,
                k_combine,
                sinc: constant(&sinc)?,
                hil: constant(&hil)?,
                weight: constant(&weight)?,
                seg: empty(),
                seg_t0: empty(),
                raw: empty(),
                a0: empty(),
                slope: empty(),
                full: empty(),
                hi: empty(),
                cov: empty(),
                out: empty(),
            };
            Ok(Self { api, ni: inner.weight.len(), state: Mutex::new(state) })
        }
    }

    /// Grow `buf` to at least `bytes`.
    unsafe fn ensure(&self, context: Handle, buf: &mut Buf, bytes: usize) -> Result<()> {
        if buf.cap >= bytes {
            return Ok(());
        }
        if !buf.mem.is_null() {
            (self.api.release_mem_object)(buf.mem);
        }
        let cap = bytes.next_power_of_two();
        let mut err = 0;
        buf.mem = (self.api.create_buffer)(context, CL_MEM_READ_WRITE, cap, null_mut(), &mut err);
        check(err, "clCreateBuffer")?;
        buf.cap = cap;
        Ok(())
    }

    unsafe fn write<T>(&self, queue: Handle, buf: &Buf, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        check(
            (self.api.enqueue_write_buffer)(queue, buf.mem, CL_TRUE, 0, std::mem::size_of_val(data), data.as_ptr().cast(), 0, null(), null_mut()),
            "clEnqueueWriteBuffer",
        )
    }

    unsafe fn arg<T>(&self, kernel: Handle, index: u32, value: &T) -> Result<()> {
        check((self.api.set_kernel_arg)(kernel, index, std::mem::size_of::<T>(), (value as *const T).cast()), "clSetKernelArg")
    }

    unsafe fn launch(&self, queue: Handle, kernel: Handle, n: usize) -> Result<()> {
        let local = 256usize;
        let global = n.div_ceil(local) * local;
        check(
            (self.api.enqueue_nd_range_kernel)(queue, kernel, 1, null(), &global, &local, 0, null(), null_mut()),
            "clEnqueueNDRangeKernel",
        )
    }

    /// Fill `out` with plan samples `m..m + out.len()`.
    pub(crate) fn run(&self, plan: &Plan, m: usize, out: &mut [f32]) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        let (len, ni) = (plan.len, self.ni);
        unsafe {
            let ctx = st.context;
            self.ensure(ctx, &mut st.seg, len * 4)?;
            self.ensure(ctx, &mut st.seg_t0, plan.seg_t0.len() * 8)?;
            self.ensure(ctx, &mut st.full, ni * len * 4)?;
            self.ensure(ctx, &mut st.hi, ni * len * 8)?;
            self.ensure(ctx, &mut st.cov, ni * len)?;
            self.ensure(ctx, &mut st.out, out.len() * 4)?;
            self.write(st.queue, &st.seg, &plan.seg)?;
            self.write(st.queue, &st.seg_t0, &plan.seg_t0)?;

            for (i, span) in plan.inputs.iter().enumerate() {
                let base = (i * len) as i32;
                let Some(span) = span else {
                    // Not covered anywhere in this read.
                    let zeros = vec![0u8; len];
                    check(
                        (self.api.enqueue_write_buffer)(st.queue, st.cov.mem, CL_TRUE, i * len, len, zeros.as_ptr().cast(), 0, null(), null_mut()),
                        "clEnqueueWriteBuffer",
                    )?;
                    continue;
                };
                self.ensure(ctx, &mut st.raw, span.raw.len() * 4)?;
                self.ensure(ctx, &mut st.a0, span.a0.len() * 8)?;
                self.ensure(ctx, &mut st.slope, span.slope.len() * 8)?;
                self.write(st.queue, &st.raw, &span.raw)?;
                self.write(st.queue, &st.a0, &span.a0)?;
                self.write(st.queue, &st.slope, &span.slope)?;
                let k = st.k_read;
                let len_i = len as i32;
                self.arg(k, 0, &st.raw.mem)?;
                self.arg(k, 1, &st.seg.mem)?;
                self.arg(k, 2, &st.seg_t0.mem)?;
                self.arg(k, 3, &st.a0.mem)?;
                self.arg(k, 4, &st.slope.mem)?;
                self.arg(k, 5, &plan.s0)?;
                self.arg(k, 6, &len_i)?;
                self.arg(k, 7, &base)?;
                self.arg(k, 8, &st.sinc)?;
                self.arg(k, 9, &st.hil)?;
                self.arg(k, 10, &st.full.mem)?;
                self.arg(k, 11, &st.hi.mem)?;
                self.arg(k, 12, &st.cov.mem)?;
                self.launch(st.queue, k, len)?;
                // The raw/a0/slope buffers are reused by the next input.
                check((self.api.finish)(st.queue), "clFinish")?;
            }

            let k = st.k_combine;
            let (ni_i, len_i, m_i, n_i) = (ni as i32, len as i32, m as i32, out.len() as i32);
            self.arg(k, 0, &ni_i)?;
            self.arg(k, 1, &len_i)?;
            self.arg(k, 2, &m_i)?;
            self.arg(k, 3, &n_i)?;
            self.arg(k, 4, &st.full.mem)?;
            self.arg(k, 5, &st.hi.mem)?;
            self.arg(k, 6, &st.cov.mem)?;
            self.arg(k, 7, &st.weight)?;
            self.arg(k, 8, &st.out.mem)?;
            self.launch(st.queue, k, out.len())?;
            check(
                (self.api.enqueue_read_buffer)(st.queue, st.out.mem, CL_TRUE, 0, out.len() * 4, out.as_mut_ptr().cast(), 0, null(), null_mut()),
                "clEnqueueReadBuffer",
            )?;
        }
        Ok(())
    }
}
