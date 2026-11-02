use anyhow::Result;
use clap::{Parser, Subcommand};
use rfx::pack::{info, pack, unpack, Device, PackOpts};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rfx", about = "Several RF captures of one disc in one lossless .rfx file")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Pack captures (the first is the primary) into one .rfx file.
    Pack {
        #[arg(short, long)]
        output: PathBuf,
        /// Predictor strength: 1 fastest to 4 strongest.
        #[arg(short, long, default_value_t = 3)]
        level: u32,
        /// Samples per independently coded segment.
        #[arg(long, default_value_t = 262144)]
        segment: usize,
        /// Skip the check that each segment decodes back to itself.
        #[arg(long)]
        no_verify: bool,
        /// Pass 0 dumps already made (pass0_<i>.txt, in the order of the inputs).
        #[arg(long)]
        pass0: Option<PathBuf>,
        /// ld-decode, to locate the lines of the captures (pass 0).
        #[arg(long, default_value = "ld-decode")]
        ld_decode: PathBuf,
        /// Capture sample rate in MHz, for pass 0.
        #[arg(long, default_value_t = 75.0)]
        rate_mhz: f64,
        /// Capture format, for pass 0.
        #[arg(long, default_value = "s16")]
        format: String,
        /// Where the predictor runs: cpu or gpu (OpenCL, same bits).
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Raw 16-bit captures.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Rebuild the original files, checking each against its SHA-256.
    Unpack {
        input: PathBuf,
        #[arg(short = 'd', long, default_value = ".")]
        outdir: PathBuf,
        /// Where the predictor runs: cpu or gpu (OpenCL, same bits).
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Extract only this source (0 is the primary).
        #[arg(long)]
        only: Option<usize>,
    },
    /// Development: compare the CPU and GPU predictors on one file.
    GpuCheck {
        /// Predict from a (synthetic) reference window too.
        #[arg(long)]
        with_ref: bool,
        input: PathBuf,
        #[arg(short, long, default_value_t = 3)]
        level: u32,
        #[arg(long, default_value_t = 8)]
        segments: usize,
        #[arg(long, default_value_t = 262144)]
        segment: usize,
    },
    /// Development: single-thread speed of each stage.
    Bench {
        input: PathBuf,
        #[arg(short, long, default_value_t = 3)]
        level: u32,
        #[arg(long, default_value_t = 16)]
        segments: usize,
        #[arg(long, default_value_t = 262144)]
        segment: usize,
    },
    /// List the sources of a .rfx file.
    Info { input: PathBuf },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Pack { output, level, segment, no_verify, pass0, ld_decode, rate_mhz, format, device, inputs } => {
            let opts = PackOpts { device: parse_device(&device)?, level, segment, verify: !no_verify, pass0_dir: pass0, ld_decode, rate_mhz, format };
            let rep = pack(&inputs, &output, &opts)?;
            let (a, b): (u64, u64) = rep.iter().fold((0, 0), |s, r| (s.0 + r.size, s.1 + r.packed));
            eprintln!("total: {a} -> {b} bytes ({:.1}%)", 100.0 * b as f64 / a.max(1) as f64);
        }
        Cmd::Unpack { input, outdir, only, device } => unpack(&input, &outdir, only, parse_device(&device)?)?,
        Cmd::Info { input } => info(&input)?,
        Cmd::Bench { input, level, segments, segment } => {
            rfx::pack::bench(&input, level, segments, segment)?;
            rfx::pack::bench_pred(&input, level, segments, segment)?
        }
        Cmd::GpuCheck { input, level, segments, segment, with_ref } => rfx::pack::gpu_check(&input, level, segments, segment, with_ref)?,
    }
    Ok(())
}

fn parse_device(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "gpu" => Ok(Device::Gpu),
        _ => anyhow::bail!("--device is cpu or gpu"),
    }
}
