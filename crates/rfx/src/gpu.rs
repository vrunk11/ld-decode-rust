//! The predictor on an OpenCL GPU, through the decoder's shared OpenCL layer.
//! Only the prediction runs there (one work-group per segment): the range
//! coder stays on the CPU, which is possible because the residual model never
//! looks at a prediction.
//!
//! Three parts, each with a command queue of its own so that one chunk can be
//! uploading while another is running and a third is coming back: an
//! [`Uploader`], a [`Runner`] and a [`Downloader`], passing a [`Dev`] (the
//! device buffers of one chunk) along. [`GpuSeg`] chains them for the callers
//! that want one call.
use crate::pred::Preset;
use anyhow::Result;
use ld_decode::opencl::{Arg, Buffer, Gpu, Kernel, Queue};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SOURCE: &str = include_str!("kernel.cl");

/// Microseconds spent so far in upload, kernel and download, all callers together.
pub static TIMES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn tick(which: usize, t: Instant) {
    TIMES[which].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// The device buffers of one chunk.
pub struct Dev {
    input: Buffer,
    refwin: Buffer,
    out: Buffer,
    lens: Buffer,
    /// Segments held.
    segs: usize,
}

pub struct Uploader {
    queue: Queue,
    gpu: Arc<Gpu>,
    seg: usize,
    rw: usize,
}

pub struct Runner {
    queue: Queue,
    kernel: Kernel,
    lam: f32,
    seg: usize,
    rw: usize,
}

pub struct Downloader {
    queue: Queue,
    seg: usize,
}

/// Build the kernel for `pre` and the three parts, for segments of `seg`
/// samples. The reference windows are `window_len` long (see [`Uploader`]).
pub fn open(pre: &Preset, use_ref: bool, seg: usize) -> Result<(Uploader, Runner, Downloader)> {
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
    let rw = seg + 2 * pre.taps;
    Ok((
        Uploader { queue: gpu.queue()?, gpu: Arc::clone(&gpu), seg, rw },
        Runner { queue: gpu.queue()?, kernel, lam: f32::from_bits(pre.lam_bits), seg, rw },
        Downloader { queue: gpu.queue()?, seg },
    ))
}

impl Uploader {
    /// Reference window length per segment (`seg + 2 * taps`).
    pub fn window_len(&self) -> usize {
        self.rw
    }

    pub fn dev(&self) -> Result<Dev> {
        Ok(Dev { input: self.gpu.buffer(1)?, refwin: self.gpu.buffer(1)?, out: self.gpu.buffer(1)?, lens: self.gpu.buffer(1)?, segs: 0 })
    }

    /// Send `lens.len()` segments of `seg` values (the tail of a short one is
    /// padding) and their reference windows (none for a source predicted from
    /// its own past only). Returns when the data is on the device.
    pub fn upload(&self, dev: &mut Dev, input: &[i32], refwin: &[f32], lens: &[i32]) -> Result<()> {
        let t = Instant::now();
        dev.segs = lens.len();
        dev.input.ensure(input.len() * 4)?;
        dev.out.ensure(input.len() * 4)?;
        dev.lens.ensure(lens.len() * 4)?;
        dev.refwin.ensure(refwin.len().max(1) * 4)?;
        self.queue.write(&dev.input, 0, input)?;
        self.queue.write(&dev.lens, 0, lens)?;
        if !refwin.is_empty() {
            self.queue.write(&dev.refwin, 0, refwin)?;
        }
        self.queue.finish()?;
        tick(0, t);
        Ok(())
    }
}

impl Runner {
    /// `mode` 0: predictions of the samples uploaded; 1: the samples whose
    /// residuals were uploaded. The result stays in `dev`.
    pub fn run(&mut self, dev: &Dev, mode: i32) -> Result<()> {
        let t = Instant::now();
        self.kernel.args(&[
            Arg::Buf(&dev.input),
            Arg::Buf(&dev.refwin),
            Arg::Buf(&dev.out),
            Arg::Buf(&dev.lens),
            Arg::I32(self.seg as i32),
            Arg::I32(self.rw as i32),
            Arg::F32(self.lam),
            Arg::I32(mode),
        ])?;
        self.queue.launch(&self.kernel, &[dev.segs * 32], 32)?;
        self.queue.finish()?;
        tick(1, t);
        Ok(())
    }
}

impl Downloader {
    pub fn download(&self, dev: &Dev) -> Result<Vec<i32>> {
        let t = Instant::now();
        let mut out = vec![0i32; dev.segs * self.seg];
        self.queue.read(&dev.out, 0, &mut out)?;
        self.queue.finish()?;
        tick(2, t);
        Ok(out)
    }
}

/// The three parts chained, for one chunk at a time.
pub struct GpuSeg {
    up: Uploader,
    run: Runner,
    down: Downloader,
    dev: Dev,
}

impl GpuSeg {
    /// Kernel for `pre`, segments of `seg` samples.
    pub fn new(pre: &Preset, use_ref: bool, seg: usize) -> Result<Self> {
        let (up, run, down) = open(pre, use_ref, seg)?;
        let dev = up.dev()?;
        Ok(Self { up, run, down, dev })
    }

    /// Reference window length per segment (`seg + 2 * taps`).
    pub fn window_len(&self) -> usize {
        self.up.window_len()
    }

    /// Upload, run and download `lens.len()` segments: see [`Uploader::upload`]
    /// and [`Runner::run`].
    pub fn run(&mut self, mode: i32, input: &[i32], refwin: &[f32], lens: &[i32]) -> Result<Vec<i32>> {
        self.up.upload(&mut self.dev, input, refwin, lens)?;
        self.run.run(&self.dev, mode)?;
        self.down.download(&self.dev)
    }

    /// Seconds spent so far in upload, kernel and download (all callers).
    pub fn times() -> [f64; 3] {
        [0, 1, 2].map(|i| TIMES[i].load(Ordering::Relaxed) as f64 / 1e6)
    }

    pub fn reset_times() {
        for t in &TIMES {
            t.store(0, Ordering::Relaxed);
        }
    }
}
