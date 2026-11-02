//! The predictor on an OpenCL GPU, through the decoder's shared OpenCL layer.
//! Only the prediction runs there (one work-group per segment): the range
//! coder stays on the CPU, which is possible because the residual model never
//! looks at a prediction.
use crate::pred::Preset;
use anyhow::Result;
use ld_decode::opencl::{Arg, Buffer, Gpu, Kernel, Queue};
use std::sync::Arc;

const SOURCE: &str = include_str!("kernel.cl");

pub struct GpuSeg {
    queue: Queue,
    kernel: Kernel,
    gpu: Arc<Gpu>,
    lam: f32,
    seg: usize,
    rw: usize,
    input: Buffer,
    refwin: Buffer,
    out: Buffer,
    lens: Buffer,
    /// Seconds spent so far moving data to the device, running, and back.
    pub t_upload: f64,
    pub t_kernel: f64,
    pub t_download: f64,
}

impl GpuSeg {
    /// Kernel for `pre`, segments of `seg` samples.
    pub fn new(pre: &Preset, use_ref: bool, seg: usize) -> Result<Self> {
        let gpu = Gpu::get()?;
        let d = pre.dim(use_ref);
        let options = format!(
            "-DOWN={} -DTAPS={} -DUSEREF={} -DSTRIDE={} -DEVERY={} -DD={} -DDP={}",
            pre.own,
            pre.taps,
            i32::from(use_ref),
            pre.stride,
            pre.every,
            d,
            pre.leaves(use_ref)
        );
        let options = if gpu.name.contains("NVIDIA") { format!("{options} -DNVIDIA=1") } else { options };
        let options = format!("{options} {}", std::env::var("RFX_GPU_DEFS").unwrap_or_default());
        eprintln!("OpenCL device {}", gpu.name);
        let kernel = gpu.kernels(SOURCE, &options, &["rfx_seg"])?.remove(0);
        Ok(Self {
            queue: gpu.queue()?,
            kernel,
            lam: f32::from_bits(pre.lam_bits),
            seg,
            rw: seg + 2 * pre.taps,
            input: gpu.buffer(1)?,
            refwin: gpu.buffer(1)?,
            out: gpu.buffer(1)?,
            lens: gpu.buffer(1)?,
            t_upload: 0.0,
            t_kernel: 0.0,
            t_download: 0.0,
            gpu,
        })
    }

    /// Reference window length per segment (`seg + 2 * taps`).
    pub fn window_len(&self) -> usize {
        self.rw
    }

    /// `mode` 0: predictions of the samples in `input`; 1: the samples whose
    /// residuals are in `input`. `input` and the result hold `lens.len()`
    /// segments of `seg` values (the tail of a short segment is padding);
    /// `refwin` holds `lens.len()` windows of `window_len()` values, or
    /// nothing for a source predicted from its own past only.
    pub fn run(&mut self, mode: i32, input: &[i32], refwin: &[f32], lens: &[i32]) -> Result<Vec<i32>> {
        let b = lens.len();
        let _ = &self.gpu;
        let t0 = std::time::Instant::now();
        self.input.ensure(input.len() * 4)?;
        self.out.ensure(input.len() * 4)?;
        self.lens.ensure(lens.len() * 4)?;
        self.refwin.ensure(refwin.len().max(1) * 4)?;
        self.queue.write(&self.input, 0, input)?;
        self.queue.write(&self.lens, 0, lens)?;
        if !refwin.is_empty() {
            self.queue.write(&self.refwin, 0, refwin)?;
        }
        self.queue.finish()?;
        let t1 = std::time::Instant::now();
        self.kernel.args(&[
            Arg::Buf(&self.input),
            Arg::Buf(&self.refwin),
            Arg::Buf(&self.out),
            Arg::Buf(&self.lens),
            Arg::I32(self.seg as i32),
            Arg::I32(self.rw as i32),
            Arg::F32(self.lam),
            Arg::I32(mode),
        ])?;
        self.queue.launch(&self.kernel, &[b * 32], 32)?;
        self.queue.finish()?;
        let t2 = std::time::Instant::now();
        let mut out = vec![0i32; input.len()];
        self.queue.read(&self.out, 0, &mut out)?;
        self.queue.finish()?;
        let t3 = std::time::Instant::now();
        self.t_upload += (t1 - t0).as_secs_f64();
        self.t_kernel += (t2 - t1).as_secs_f64();
        self.t_download += (t3 - t2).as_secs_f64();
        Ok(out)
    }
}
