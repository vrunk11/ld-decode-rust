use anyhow::Result;
use clap::{Parser, Subcommand};
use rfx::pack::{info, pack, unpack, Device, OrderMode, PackOpts};
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
        /// Capture format: auto (from the extension), s16, r16, s8, r8, lds, or flac / ldf (decoded
        /// with ffmpeg to raw 16-bit samples first). Also the format given to pass 0.
        #[arg(long, default_value = "auto")]
        format: String,
        /// Where the predictor runs: cpu or gpu (OpenCL, same bits).
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Order of coding: `auto` puts the cleanest capture of each programme first, `given` keeps the command line order.
        #[arg(long, default_value = "auto")]
        order: String,
        /// Code the tracks as FLAC instead of with the predictor: the file is then an ordinary
        /// Matroska audio file (a FLAC track per capture) that audio programs open, several
        /// times larger; unpack gives the captures back all the same.
        #[arg(long)]
        flac: bool,
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
        /// Extract only this track (0 is the primary).
        #[arg(long)]
        only: Option<usize>,
        /// Write the captures in another format (s16, r16, s8, r8, lds, flac, ldf, or mflac: the tracks as the channels of FLAC files, 8 a file, to open in an audio program) instead of the
        /// one they came in; the samples are checked by hash.
        #[arg(long = "as")]
        as_format: Option<String>,
        /// Write only the metadata (each track's .yaml as it was, and <archive>.yaml with all of it);
        /// no sample is decoded.
        #[arg(long)]
        meta: bool,
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
    /// Development: how noisy each capture is (cost of coding it alone, on the same frames).
    Noise {
        /// Pass 0 dumps, in the order of the inputs.
        #[arg(long)]
        pass0: PathBuf,
        #[arg(long, default_value_t = 8)]
        windows: usize,
        #[arg(long, default_value_t = 262144)]
        window: usize,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Development: would other references have cost less for one capture?
    Refsel {
        /// Coding place of the target (0 is the primary).
        #[arg(long)]
        target: usize,
        #[arg(long, default_value_t = 12)]
        segments: usize,
        #[arg(long)]
        pass0: PathBuf,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Development: experiments on predictor and model (see lab.rs).
    Lab {
        /// model, tables or spectrum
        what: String,
        #[arg(long)]
        target: usize,
        #[arg(long, default_value_t = 12)]
        segments: usize,
        #[arg(long)]
        pass0: PathBuf,
        /// Number of tones for the tones experiment.
        #[arg(long, default_value_t = 16)]
        tones: usize,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Merge .rfx files into one (the tracks of each after the ones of the
    /// previous; no track is coded again unless `--recode` is given).
    Merge {
        #[arg(short, long)]
        output: PathBuf,
        /// Code the tracks of the later files again against everything before
        /// them, so that captures of one disc in different files predict each other.
        #[arg(long)]
        recode: bool,
        #[command(flatten)]
        work: WorkOpts,
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,
    },
    /// Write some of the tracks of a .rfx file as a .rfx file of their own.
    Extract {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Tracks to keep (0 is the primary), e.g. 0,3,5.
        #[arg(long, required = true, value_delimiter = ',')]
        only: Vec<usize>,
        #[command(flatten)]
        work: WorkOpts,
    },
    /// Write a .rfx file without some of the tracks of another.
    Remove {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Tracks to remove (0 is the primary), e.g. 2,4.
        #[arg(long, required = true, value_delimiter = ',')]
        track: Vec<usize>,
        #[command(flatten)]
        work: WorkOpts,
    },
    /// Write a .rfx file with more captures than another.
    Add {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        work: WorkOpts,
        /// Captures to add, in any format `pack` takes.
        #[arg(required = true)]
        captures: Vec<PathBuf>,
    },
    /// List the tracks of a .rfx file (nothing is decoded).
    Info {
        input: PathBuf,
        /// Print the whole metadata as YAML instead of the summary.
        #[arg(long)]
        yaml: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Pack { output, level, segment, no_verify, pass0, ld_decode, rate_mhz, format, device, order, flac, inputs } => {
            let opts = PackOpts { order: parse_order(&order)?, device: parse_device(&device)?, level, segment, verify: !no_verify, pass0_dir: pass0, ld_decode, rate_mhz, format };
            if flac {
                return rfx::flacmode::pack(&inputs, &output, &opts);
            }
            let rep = pack(&inputs, &output, &opts)?;
            let (a, b): (u64, u64) = rep.iter().fold((0, 0), |s, r| (s.0 + r.size, s.1 + r.packed));
            eprintln!("total: {a} -> {b} bytes ({:.1}%)", 100.0 * b as f64 / a.max(1) as f64);
        }
        Cmd::Unpack { input, outdir, only, device, as_format, meta } => {
            if rfx::flacmode::is_flac_archive(&input) {
                if meta {
                    return rfx::flacmode::unpack_meta(&input, &outdir, only);
                }
                let target = as_format.map(|f| rfx::pack::OutFmt::parse(&f).ok_or_else(|| anyhow::anyhow!("--as: unknown format {f} for a FLAC-mode archive"))).transpose()?;
                return rfx::flacmode::unpack(&input, &outdir, only, target);
            }
            if meta {
                return rfx::pack::unpack_meta(&input, &outdir, only);
            }
            if as_format.as_deref() == Some("mflac") {
                return rfx::mflac::export(&input, &outdir, only);
            }
            let target = as_format.map(|f| rfx::pack::OutFmt::parse(&f).ok_or_else(|| anyhow::anyhow!("--as: unknown format {f}"))).transpose()?;
            unpack(&input, &outdir, only, parse_device(&device)?, target)?
        }
        Cmd::Info { input, yaml } => {
            if rfx::flacmode::is_flac_archive(&input) {
                if yaml {
                    print!("{}", rfx::flacmode::meta_yaml(&input)?);
                    return Ok(());
                }
                return rfx::flacmode::info(&input);
            }
            if yaml {
                print!("{}", rfx::pack::meta_yaml(&input)?);
            } else {
                info(&input)?
            }
        }
        Cmd::Merge { output, recode, work, inputs } => {
            let picks = rfx::edit::pick_all(&inputs)?;
            rfx::edit::assemble(&inputs, &picks, &output, &work_opts(&work)?, recode.then_some(1))?
        }
        Cmd::Extract { input, output, only, work } => {
            let picks = rfx::edit::pick_some(&input, &only)?;
            rfx::edit::assemble(&[input], &picks, &output, &work_opts(&work)?, None)?
        }
        Cmd::Remove { input, output, track, work } => {
            let picks = rfx::edit::pick_without(&input, &track)?;
            rfx::edit::assemble(&[input], &picks, &output, &work_opts(&work)?, None)?
        }
        Cmd::Add { input, output, captures, work } => {
            let mut picks = rfx::edit::pick_all(std::slice::from_ref(&input))?;
            picks.extend(captures.into_iter().map(rfx::edit::Pick::New));
            rfx::edit::assemble(&[input], &picks, &output, &work_opts(&work)?, None)?
        }
        Cmd::Lab { what, target, segments, pass0, tones, inputs } => {
            let opts = PackOpts { order: OrderMode::Given, device: Device::Cpu, level: 3, segment: 262144, verify: false, pass0_dir: Some(pass0), ld_decode: PathBuf::from("ld-decode"), rate_mhz: 75.0, format: "auto".into() };
            match what.as_str() {
                "model" => rfx::labmodel::model(&inputs, &opts, target, segments)?,
                "tables" => rfx::labtables::tables(&inputs, &opts)?,
                "spectrum" => rfx::labspec::spectrum(&inputs, &opts, target, segments)?,
                "cascade" => rfx::labspec::cascade(&inputs, &opts, target, segments)?,
                "tones" => rfx::labtones::tones(&inputs, &opts, target, segments, tones)?,
                "residual" => rfx::labresid::residual(&inputs, &opts, target, segments)?,
                "precision" => rfx::labprec::precision(&inputs, &opts, target, segments)?,
                other => anyhow::bail!("unknown experiment {other}"),
            }
        }
        Cmd::Refsel { target, segments, pass0, inputs } => {
            let opts = PackOpts { order: OrderMode::Given, device: Device::Cpu, level: 3, segment: 262144, verify: false, pass0_dir: Some(pass0), ld_decode: PathBuf::from("ld-decode"), rate_mhz: 75.0, format: "auto".into() };
            rfx::pack::refsel(&inputs, &opts, target, segments)?
        }
        Cmd::Noise { pass0, windows, window, inputs } => {
            let fields: Vec<_> = (0..inputs.len()).map(|i| rfx::align::parse_pass0(&pass0.join(format!("pass0_{i}.txt")))).collect::<Result<_>>()?;
            let refs: Vec<&[rfx::align::Field]> = fields.iter().map(|f| f.as_slice()).collect();
            let fmts: Vec<rfx::fmt::Fmt> = inputs.iter().map(|p| rfx::fmt::Fmt::from_path(p).unwrap_or(rfx::fmt::Fmt::S16)).collect();
            let sc = rfx::order::scores(&inputs, &fmts, &refs, windows, window)?;
            println!("{:<34} {:>9} {:>8} {:>8} {:>7} {:>4}", "capture", "bits/smp", "min", "max", "rms", "win");
            for (p, s) in inputs.iter().zip(&sc) {
                println!("{:<34} {:>9.3} {:>8.3} {:>8.3} {:>7.1} {:>4}", p.file_name().unwrap().to_string_lossy().trim_end_matches(".ddd.s16"), s.bits, s.min, s.max, s.rms, s.windows);
            }
        }
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

fn parse_order(s: &str) -> Result<OrderMode> {
    match s {
        "auto" => Ok(OrderMode::Auto),
        "given" => Ok(OrderMode::Given),
        _ => anyhow::bail!("--order is auto or given"),
    }
}

/// What the editing commands need to code or decode a track again.
#[derive(clap::Args, Clone, Debug)]
struct WorkOpts {
    /// Predictor strength for tracks coded again or added: 1 fastest to 4 strongest.
    #[arg(short, long, default_value_t = 3)]
    level: u32,
    /// Skip the check that each coded segment decodes back to itself.
    #[arg(long)]
    no_verify: bool,
    /// Pass 0 dumps already made (pass0_<i>.txt, in the order of the tracks written).
    #[arg(long)]
    pass0: Option<PathBuf>,
    /// ld-decode, to locate the lines of the captures (pass 0).
    #[arg(long, default_value = "ld-decode")]
    ld_decode: PathBuf,
    /// Capture sample rate in MHz, for pass 0.
    #[arg(long, default_value_t = 75.0)]
    rate_mhz: f64,
    /// Where the predictor runs: cpu or gpu (OpenCL, same bits).
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn work_opts(w: &WorkOpts) -> Result<PackOpts> {
    Ok(PackOpts { order: OrderMode::Given, device: parse_device(&w.device)?, level: w.level, segment: 262144, verify: !w.no_verify, pass0_dir: w.pass0.clone(), ld_decode: w.ld_decode.clone(), rate_mhz: w.rate_mhz, format: "auto".into() })
}
