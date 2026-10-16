//! LDX output: the decode written as one Matroska file, or one stream on
//! stdout, following ld-decode-tools' `docs/Specifications/container-format.md`
//! (profile 1). The `tbc` and `pcm` tracks are raw or FLAC (`--ldx-codec`);
//! the metadata and EFM tracks are always raw:
//!
//! - `tbc` (track 1): one field per block, the bytes of the `.tbc` file;
//! - `metadata` (track 2): ld-decode JSON. The parameters go first (no
//!   fields), then each field's entry just before its video (as a stream must
//!   send them), and at the end the whole set, identical to the `.tbc.json`, so
//!   that the current metadata is exactly what the JSON sidecar would hold;
//! - `pcm` (analogue audio, 44.1 kHz stereo) and `efm` (t-values, 64 KiB
//!   chunks), when they are decoded.
//!
//! Only what the profile uses of Matroska is written, by hand: the EBML header,
//! a segment (sized for a file, of unknown size for a stream), Info, Tracks, a
//! 4 KiB Void after the tracks (files only), the LDX tags, clusters of
//! BlockGroups spanning less than 30 ms, and one cue per block at the end of a
//! file.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{json, Value};

const PROFILE: u64 = 1;
/// One tick of a block time is a microsecond.
const TIMESTAMP_SCALE_NS: u64 = 1000;
/// A cluster is closed when a block is this far after its start.
const CLUSTER_SPAN_US: i64 = 30_000;
/// Room after the tracks, so a track can be added to a finished file in place.
const TRACK_PADDING_BYTES: usize = 4096;
/// Size of the EFM (byte) chunks.
const BYTE_CHUNK: usize = 65536;
/// The audio sampling frequency of a raw sample track: a stand-in, the real
/// rate is in the track's parameters.
const STAND_IN_RATE: f64 = 48000.0;
/// The analogue audio rate of the `.pcm` output.
const PCM_RATE: i128 = 44100;
const APP: &str = "ld-decode-tools ldx";

/// Matroska element ids.
mod id {
    pub const EBML: u32 = 0x1A45_DFA3;
    pub const EBML_VERSION: u32 = 0x4286;
    pub const EBML_READ_VERSION: u32 = 0x42F7;
    pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
    pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
    pub const DOC_TYPE: u32 = 0x4282;
    pub const DOC_TYPE_VERSION: u32 = 0x4287;
    pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;
    pub const SEGMENT: u32 = 0x1853_8067;
    pub const INFO: u32 = 0x1549_A966;
    pub const TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
    pub const MUXING_APP: u32 = 0x4D80;
    pub const WRITING_APP: u32 = 0x5741;
    pub const TRACKS: u32 = 0x1654_AE6B;
    pub const TRACK_ENTRY: u32 = 0xAE;
    pub const TRACK_NUMBER: u32 = 0xD7;
    pub const TRACK_UID: u32 = 0x73C5;
    pub const TRACK_TYPE: u32 = 0x83;
    pub const FLAG_LACING: u32 = 0x9C;
    pub const TRACK_NAME: u32 = 0x536E;
    pub const CODEC_ID: u32 = 0x86;
    pub const CODEC_PRIVATE: u32 = 0x63A2;
    pub const AUDIO: u32 = 0xE1;
    pub const SAMPLING_FREQUENCY: u32 = 0xB5;
    pub const CHANNELS: u32 = 0x9F;
    pub const BIT_DEPTH: u32 = 0x6264;
    pub const VOID: u32 = 0xEC;
    pub const TAGS: u32 = 0x1254_C367;
    pub const TAG: u32 = 0x7373;
    pub const TARGETS: u32 = 0x63C0;
    pub const TAG_TRACK_UID: u32 = 0x63C5;
    pub const SIMPLE_TAG: u32 = 0x67C8;
    pub const TAG_NAME: u32 = 0x45A3;
    pub const TAG_STRING: u32 = 0x4487;
    pub const CLUSTER: u32 = 0x1F43_B675;
    pub const TIMESTAMP: u32 = 0xE7;
    pub const BLOCK_GROUP: u32 = 0xA0;
    pub const BLOCK: u32 = 0xA1;
    pub const CUES: u32 = 0x1C53_BB6B;
    pub const CUE_POINT: u32 = 0xBB;
    pub const CUE_TIME: u32 = 0xB3;
    pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
    pub const CUE_TRACK: u32 = 0xF7;
    pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
    pub const CUE_RELATIVE_POSITION: u32 = 0xF0;
}

// ---------------------------------------------------------------------------
// EBML encoding

fn id_bytes(id: u32) -> Vec<u8> {
    let b = id.to_be_bytes();
    let skip = b.iter().position(|&x| x != 0).unwrap_or(3);
    b[skip..].to_vec()
}

/// Smallest EBML variable-size integer holding `v` (the all-ones value of each
/// length is reserved for "unknown").
fn vint(v: u64) -> Vec<u8> {
    let n = (1..=8usize).find(|&n| v < (1u64 << (7 * n)) - 1).unwrap_or(8);
    let mut out = vec![0u8; n];
    for (i, byte) in out.iter_mut().rev().enumerate() {
        *byte = (v >> (8 * i)) as u8;
    }
    out[0] |= 0x80 >> (n - 1);
    out
}

fn header(id: u32, size: u64) -> Vec<u8> {
    let mut h = id_bytes(id);
    h.extend(vint(size));
    h
}

fn element(id: u32, data: &[u8]) -> Vec<u8> {
    let mut e = header(id, data.len() as u64);
    e.extend_from_slice(data);
    e
}

fn uint(id: u32, v: u64) -> Vec<u8> {
    let b = v.to_be_bytes();
    let skip = b.iter().position(|&x| x != 0).unwrap_or(7);
    element(id, &b[skip..])
}

fn string(id: u32, s: &str) -> Vec<u8> {
    element(id, s.as_bytes())
}

fn float(id: u32, v: f64) -> Vec<u8> {
    element(id, &v.to_be_bytes())
}

fn master(id: u32, children: &[Vec<u8>]) -> Vec<u8> {
    element(id, &children.concat())
}

/// `v` as an EBML variable-size integer of exactly `n` bytes.
fn vint_fixed(v: u64, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    for (i, byte) in out.iter_mut().rev().enumerate() {
        *byte = (v >> (8 * i)) as u8;
    }
    out[0] |= 0x80 >> (n - 1);
    out
}

/// A signed EBML integer (the size differences of EBML lacing): the value plus
/// the bias of the shortest length that holds it.
fn signed_vint(d: i64) -> Vec<u8> {
    let n = (1..=8usize)
        .find(|&n| d.unsigned_abs() < (1u64 << (7 * n - 1)))
        .unwrap_or(8);
    let bias = (1i64 << (7 * n - 1)) - 1;
    vint_fixed((d + bias) as u64, n)
}

/// The lace header of a block of several frames (EBML lacing): the count of
/// frames less one, the size of the first, then the difference of each next
/// size from the one before (the last size is what is left of the block).
fn ebml_lace_header(frames: &[Vec<u8>]) -> Vec<u8> {
    let mut out = vec![(frames.len() - 1) as u8];
    out.extend(vint(frames[0].len() as u64));
    for pair in frames[..frames.len() - 1].windows(2) {
        out.extend(signed_vint(pair[1].len() as i64 - pair[0].len() as i64));
    }
    out
}

// ---------------------------------------------------------------------------
// Codecs

/// How a sample track (`tbc`, `pcm`) is coded in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Raw,
    Flac,
}

impl Codec {
    fn name(self) -> &'static str {
        match self {
            Codec::Raw => "raw",
            Codec::Flac => "flac",
        }
    }
}

/// `--ldx-codec`: the codec of the video track, chosen by what the format
/// allows (spec section 6): 16-bit `tbc` is raw or FLAC, FFV1 is only for the
/// 10-bit `cvbs` frames this decoder does not write. The analogue audio follows
/// the video's choice (raw or FLAC), as `ld-container --codec` does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum CodecChoice {
    /// Raw samples (every reader reads them)
    #[default]
    Raw,
    /// FLAC (lossless, about 70 % of raw)
    Flac,
    /// FFV1 (only for 10-bit cvbs; refused for a tbc track)
    Ffv1,
    /// The best the format allows: FLAC for tbc and audio
    Auto,
}

impl CodecChoice {
    /// The codec of the `tbc` track (and of the `pcm` track with it).
    pub fn for_tbc(self) -> Result<Codec> {
        match self {
            CodecChoice::Raw => Ok(Codec::Raw),
            CodecChoice::Flac | CodecChoice::Auto => Ok(Codec::Flac),
            CodecChoice::Ffv1 => anyhow::bail!(
                "FFV1 is for the 10-bit frames of a cvbs track (LDX section 6); \
                 a tbc track is raw or flac (--ldx-codec raw|flac|auto)"
            ),
        }
    }
}

/// FLAC as LDX carries it (section 6.1): frames of 32,768 samples, at most 8 to
/// a unit, each unit coded by an encoder of its own (so a block decodes alone),
/// the stand-in rate of 48 kHz, 16-bit samples.
mod flac {
    use anyhow::{anyhow, ensure, Context, Result};
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    pub const FRAME: usize = 32_768;
    const MAX_FRAMES: usize = 8;
    const RATE: usize = 48_000;

    fn stream(samples: &[i32], channels: usize) -> Result<flacenc::component::Stream> {
        let mut encoder = flacenc::config::Encoder::default();
        encoder.block_size = FRAME;
        let config = encoder
            .into_verified()
            .map_err(|_| anyhow!("the FLAC encoder configuration was refused"))?;
        let source = flacenc::source::MemSource::from_samples(samples, channels, 16, RATE);
        flacenc::encode_with_fixed_block_size(&config, source, FRAME)
            .map_err(|e| anyhow!("FLAC encoding failed: {e:?}"))
    }

    /// The FLAC frames of one unit (`samples` interleaved, `channels` of them).
    pub fn encode(samples: &[i32], channels: usize) -> Result<Vec<Vec<u8>>> {
        ensure!(
            samples.len() / channels <= FRAME * MAX_FRAMES,
            "a FLAC unit of LDX is at most {} samples per channel",
            FRAME * MAX_FRAMES
        );
        let stream = stream(samples, channels)?;
        let mut frames = Vec::with_capacity(stream.frame_count());
        for n in 0..stream.frame_count() {
            let mut sink = flacenc::bitsink::ByteSink::new();
            stream
                .frame(n)
                .context("a FLAC frame is missing")?
                .write(&mut sink)
                .ok()
                .context("writing a FLAC frame")?;
            frames.push(sink.into_inner());
        }
        Ok(frames)
    }

    /// The `CodecPrivate` of a FLAC track: `fLaC` and the STREAMINFO block
    /// (marked the last metadata block), from an encoder of the track's shape.
    /// Encoded from a full frame and a short one, so that the block sizes it
    /// declares (16 to 32,768) cover the short last frame of every unit.
    pub fn codec_private(channels: usize) -> Result<Vec<u8>> {
        let stream = stream(&vec![0i32; (FRAME + 16) * channels], channels)?;
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream.write(&mut sink).ok().context("writing the FLAC header")?;
        let bytes = sink.into_inner();
        ensure!(bytes.len() >= 42 && &bytes[..4] == b"fLaC", "unexpected FLAC stream header");
        let mut private = bytes[..42].to_vec();
        private[4] |= 0x80;
        Ok(private)
    }
}

// ---------------------------------------------------------------------------
// Tracks

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Tbc,
    Pcm,
    Efm,
    Metadata,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Tbc => "tbc",
            Kind::Pcm => "pcm",
            Kind::Efm => "efm",
            Kind::Metadata => "metadata",
        }
    }

    fn depth(self) -> u32 {
        match self {
            Kind::Tbc | Kind::Pcm => 16,
            Kind::Efm => 8,
            Kind::Metadata => 0,
        }
    }
}

pub struct Track {
    pub kind: Kind,
    pub codec: Codec,
    pub number: u32,
    pub uid: u64,
    pub name: String,
    pub parameters: Value,
    /// For a FLAC track: `fLaC` and its STREAMINFO (the `CodecPrivate`).
    pub flac_private: Vec<u8>,
}

/// The description of a track (its `LDX_TRACK` tag, and the `CodecPrivate` of
/// a raw track).
fn description(track: &Track) -> String {
    json!({
        "ldx": PROFILE,
        "kind": track.kind.name(),
        "codec": track.codec.name(),
        "depth": track.kind.depth(),
        "parameters": track.parameters.clone(),
    })
    .to_string()
}

fn track_entry(track: &Track) -> Vec<u8> {
    let flac = track.codec == Codec::Flac;
    let mut children = vec![
        uint(id::TRACK_NUMBER, u64::from(track.number)),
        uint(id::TRACK_UID, track.uid),
        uint(id::FLAG_LACING, u64::from(flac)),
    ];
    if !track.name.is_empty() {
        children.push(string(id::TRACK_NAME, &track.name));
    }
    match track.kind {
        Kind::Tbc | Kind::Pcm => {
            // Raw samples ride in an audio track (ffmpeg keeps 16-bit PCM in
            // Matroska bit for bit).
            let channels = if track.kind == Kind::Pcm { 2 } else { 1 };
            children.push(uint(id::TRACK_TYPE, 2));
            children.push(string(id::CODEC_ID, if flac { "A_FLAC" } else { "A_PCM/INT/LIT" }));
            children.push(master(
                id::AUDIO,
                &[
                    float(id::SAMPLING_FREQUENCY, STAND_IN_RATE),
                    uint(id::CHANNELS, channels),
                    uint(id::BIT_DEPTH, 16),
                ],
            ));
        }
        Kind::Efm | Kind::Metadata => {
            children.push(uint(id::TRACK_TYPE, 33));
            children.push(string(
                id::CODEC_ID,
                &format!("D_LDX/{}", track.kind.name().to_ascii_uppercase()),
            ));
        }
    }
    // A FLAC track gives its decoder the stream header; the description of
    // every track is in its tag as well.
    if flac {
        children.push(element(id::CODEC_PRIVATE, &track.flac_private));
    } else {
        children.push(element(id::CODEC_PRIVATE, description(track).as_bytes()));
    }
    master(id::TRACK_ENTRY, &children)
}

fn simple_tag(name: &str, value: &str) -> Vec<u8> {
    master(id::SIMPLE_TAG, &[string(id::TAG_NAME, name), string(id::TAG_STRING, value)])
}

/// A random, non-zero track UID.
fn new_uid() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut z = now
        ^ (u64::from(std::process::id()) << 32)
        ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

// ---------------------------------------------------------------------------
// The Matroska writer

enum Sink {
    File(BufWriter<File>),
    Stream(Box<dyn Write + Send>),
}

struct ClusterBlock {
    track: u32,
    us: i64,
    /// The frames of the unit: one, or the frames of a codec (laced).
    frames: Vec<Vec<u8>>,
}

struct Cue {
    us: i64,
    track: u32,
    cluster: u64,
    relative: u64,
}

pub struct LdxWriter {
    sink: Sink,
    /// Bytes written so far.
    pos: u64,
    /// Where the data of the segment start (positions in cues are from here).
    segment_data: u64,
    cluster: Vec<ClusterBlock>,
    cluster_us: i64,
    cues: Vec<Cue>,
    last_us: i64,
}

impl LdxWriter {
    /// A finished file: sized segment, Void after the tracks, cues at the end.
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Self::with_sink(Sink::File(BufWriter::with_capacity(8 << 20, file))))
    }

    /// A stream (a pipe): unknown-size segment, no cues, nothing written again.
    pub fn stream(out: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::Stream(out))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            sink,
            pos: 0,
            segment_data: 0,
            cluster: Vec::new(),
            cluster_us: 0,
            cues: Vec::new(),
            last_us: 0,
        }
    }

    fn streaming(&self) -> bool {
        matches!(self.sink, Sink::Stream(_))
    }

    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        match &mut self.sink {
            Sink::File(f) => f.write_all(bytes)?,
            Sink::Stream(s) => s.write_all(bytes)?,
        }
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// Everything before the data: header, segment head, info, tracks, tags.
    pub fn begin(&mut self, tracks: &[Track], history: &Value) -> Result<()> {
        let ebml = master(
            id::EBML,
            &[
                uint(id::EBML_VERSION, 1),
                uint(id::EBML_READ_VERSION, 1),
                uint(id::EBML_MAX_ID_LENGTH, 4),
                uint(id::EBML_MAX_SIZE_LENGTH, 8),
                string(id::DOC_TYPE, "matroska"),
                uint(id::DOC_TYPE_VERSION, 4),
                uint(id::DOC_TYPE_READ_VERSION, 2),
            ],
        );
        self.put(&ebml)?;
        // An 8-byte size, "unknown" until a file is finished (then patched).
        let mut segment = id_bytes(id::SEGMENT);
        segment.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        self.put(&segment)?;
        self.segment_data = self.pos;

        let info = master(
            id::INFO,
            &[
                uint(id::TIMESTAMP_SCALE, TIMESTAMP_SCALE_NS),
                string(id::MUXING_APP, APP),
                string(id::WRITING_APP, APP),
            ],
        );
        self.put(&info)?;
        let entries: Vec<Vec<u8>> = tracks.iter().map(track_entry).collect();
        self.put(&master(id::TRACKS, &entries))?;
        if !self.streaming() {
            let mut void = header(id::VOID, TRACK_PADDING_BYTES as u64);
            void.resize(void.len() + TRACK_PADDING_BYTES, 0);
            self.put(&void)?;
        }

        let next_id = tracks.iter().map(|t| t.number).max().unwrap_or(0) + 1;
        let mut tags = vec![master(
            id::TAG,
            &[
                master(id::TARGETS, &[]),
                simple_tag("LDX_PROFILE", &PROFILE.to_string()),
                simple_tag("LDX_HISTORY", &history.to_string()),
                simple_tag("LDX_NEXT_TRACK_ID", &next_id.to_string()),
            ],
        )];
        for track in tracks {
            tags.push(master(
                id::TAG,
                &[
                    master(id::TARGETS, &[uint(id::TAG_TRACK_UID, track.uid)]),
                    simple_tag("LDX_TRACK", &description(track)),
                ],
            ));
        }
        self.put(&master(id::TAGS, &tags))
    }

    /// One block (one unit of `track`). Blocks must come in time order; a
    /// time earlier than the last one is written at the last one.
    pub fn write_block(&mut self, track: u32, time_ns: i64, data: Vec<u8>) -> Result<()> {
        self.write_frames(track, time_ns, vec![data])
    }

    /// One block made of the frames a codec gave for one unit (EBML-laced
    /// when there are several).
    pub fn write_frames(&mut self, track: u32, time_ns: i64, frames: Vec<Vec<u8>>) -> Result<()> {
        anyhow::ensure!(!frames.is_empty() && frames.len() <= 256, "a block holds 1 to 256 frames");
        let us = (time_ns + 500).div_euclid(1000).max(self.last_us);
        if !self.cluster.is_empty() && us - self.cluster_us >= CLUSTER_SPAN_US {
            self.flush_cluster()?;
        }
        if self.cluster.is_empty() {
            self.cluster_us = us;
        }
        self.cluster.push(ClusterBlock { track, us, frames });
        self.last_us = us;
        Ok(())
    }

    fn flush_cluster(&mut self) -> Result<()> {
        if self.cluster.is_empty() {
            return Ok(());
        }
        let blocks = std::mem::take(&mut self.cluster);
        let timestamp = uint(id::TIMESTAMP, self.cluster_us as u64);
        // Each block's BlockGroup header and Block header (with the track,
        // the 16-bit relative time and the flags), sized before writing.
        let mut heads = Vec::with_capacity(blocks.len());
        let mut data_size = timestamp.len() as u64;
        for b in &blocks {
            let mut prefix = vint(u64::from(b.track));
            prefix.extend_from_slice(&((b.us - self.cluster_us) as i16).to_be_bytes());
            if b.frames.len() > 1 {
                prefix.push(0x06); // EBML lacing
                prefix.extend(ebml_lace_header(&b.frames));
            } else {
                prefix.push(0x00);
            }
            let data_len: usize = b.frames.iter().map(Vec::len).sum();
            let block_len = (prefix.len() + data_len) as u64;
            let mut block_head = header(id::BLOCK, block_len);
            let group_len = block_head.len() as u64 + block_len;
            let group_head = header(id::BLOCK_GROUP, group_len);
            block_head.extend_from_slice(&prefix);
            data_size += group_head.len() as u64 + group_len;
            heads.push((group_head, block_head));
        }
        let cluster_pos = self.pos - self.segment_data;
        self.put(&header(id::CLUSTER, data_size))?;
        self.put(&timestamp)?;
        let mut relative = timestamp.len() as u64;
        let streaming = self.streaming();
        for (b, (group_head, block_head)) in blocks.iter().zip(&heads) {
            if !streaming {
                self.cues.push(Cue { us: b.us, track: b.track, cluster: cluster_pos, relative });
            }
            self.put(group_head)?;
            self.put(block_head)?;
            let mut data_len = 0;
            for frame in &b.frames {
                self.put(frame)?;
                data_len += frame.len();
            }
            relative += (group_head.len() + block_head.len() + data_len) as u64;
        }
        Ok(())
    }

    /// The last cluster, and for a file the cues and the size of the segment.
    pub fn finish(mut self) -> Result<()> {
        self.flush_cluster()?;
        if !self.streaming() && !self.cues.is_empty() {
            let points: Vec<Vec<u8>> = self
                .cues
                .iter()
                .map(|c| {
                    master(
                        id::CUE_POINT,
                        &[
                            uint(id::CUE_TIME, c.us as u64),
                            master(
                                id::CUE_TRACK_POSITIONS,
                                &[
                                    uint(id::CUE_TRACK, u64::from(c.track)),
                                    uint(id::CUE_CLUSTER_POSITION, c.cluster),
                                    uint(id::CUE_RELATIVE_POSITION, c.relative),
                                ],
                            ),
                        ],
                    )
                })
                .collect();
            self.put(&master(id::CUES, &points))?;
        }
        let size = self.pos - self.segment_data;
        let size_at = self.segment_data - 8;
        match &mut self.sink {
            Sink::File(f) => {
                f.flush()?;
                let file = f.get_mut();
                let mut b = size.to_be_bytes();
                b[0] = 0x01; // an 8-byte size (< 2^56)
                file.seek(SeekFrom::Start(size_at))?;
                file.write_all(&b)?;
                file.flush()?;
            }
            Sink::Stream(s) => s.flush()?,
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The decode as LDX tracks

/// What the `tbc` track says of the video.
pub struct VideoGeometry {
    pub system: String,
    pub field_width: usize,
    pub field_height: usize,
    pub sample_rate: f64,
}

pub struct LdxOutputConfig {
    /// Name of the capture (the video track's name).
    pub name: String,
    pub pcm: bool,
    pub efm: bool,
    /// Codec of the `tbc` track and of the `pcm` track.
    pub codec: Codec,
    /// The `LDX_HISTORY` array (see [`history`]).
    pub history: Value,
}

const TBC_TRACK: u32 = 1;
const METADATA_TRACK: u32 = 2;

struct Pending {
    ns: i64,
    seq: u64,
    track: u32,
    frames: Vec<Vec<u8>>,
}

/// Little-endian bytes of 16-bit samples.
fn le_bytes<T: Copy>(values: &[T], to: impl Fn(T) -> [u8; 2]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &v in values {
        bytes.extend_from_slice(&to(v));
    }
    bytes
}

pub struct LdxOutput {
    writer: Option<LdxWriter>,
    config: LdxOutputConfig,
    begun: bool,
    pal: bool,
    pcm_track: Option<u32>,
    efm_track: Option<u32>,
    pending: Vec<Pending>,
    seq: u64,
    fields: u64,
    last_ns: i64,
    pcm_samples: u64,
    efm_buf: Vec<u8>,
}

impl LdxOutput {
    pub fn new(writer: LdxWriter, config: LdxOutputConfig) -> Self {
        let pcm_track = config.pcm.then_some(3);
        let efm_track = config.efm.then_some(if config.pcm { 4 } else { 3 });
        Self {
            writer: Some(writer),
            config,
            begun: false,
            pal: false,
            pcm_track,
            efm_track,
            pending: Vec::new(),
            seq: 0,
            fields: 0,
            last_ns: 0,
            pcm_samples: 0,
            efm_buf: Vec::new(),
        }
    }

    pub fn begun(&self) -> bool {
        self.begun
    }

    /// Write the head of the file and queue the first metadata block: the
    /// parameters alone (`header_json` has `videoParameters` and no `fields`).
    pub fn begin(&mut self, video: &VideoGeometry, header_json: Vec<u8>) -> Result<()> {
        self.pal = video.system == "PAL";
        let codec = self.config.codec;
        let (tbc_private, pcm_private) = match codec {
            Codec::Flac => (flac::codec_private(1)?, flac::codec_private(2)?),
            Codec::Raw => (Vec::new(), Vec::new()),
        };
        let name = &self.config.name;
        let mut tracks = vec![
            Track {
                kind: Kind::Tbc,
                codec,
                flac_private: tbc_private,
                number: TBC_TRACK,
                uid: new_uid(),
                name: name.clone(),
                parameters: json!({
                    "system": video.system,
                    "sampleRate": video.sample_rate,
                    "fieldWidth": video.field_width,
                    "fieldHeight": video.field_height,
                }),
            },
            Track {
                kind: Kind::Metadata,
                codec: Codec::Raw,
                flac_private: Vec::new(),
                number: METADATA_TRACK,
                uid: new_uid(),
                name: format!("{name} metadata"),
                parameters: json!({
                    "namespace": "ldx.tbcmetadata",
                    "format": "ld-decode-json",
                    "track": TBC_TRACK,
                }),
            },
        ];
        if let Some(number) = self.pcm_track {
            tracks.push(Track {
                kind: Kind::Pcm,
                codec,
                flac_private: pcm_private,
                number,
                uid: new_uid(),
                name: format!("{name} audio"),
                parameters: json!({ "sampleRate": PCM_RATE as u64, "channels": 2, "track": TBC_TRACK }),
            });
        }
        if let Some(number) = self.efm_track {
            tracks.push(Track {
                kind: Kind::Efm,
                codec: Codec::Raw,
                flac_private: Vec::new(),
                number,
                uid: new_uid(),
                name: format!("{name} efm"),
                parameters: json!({ "track": TBC_TRACK }),
            });
        }
        let writer = self.writer.as_mut().context("LDX output already finished")?;
        writer.begin(&tracks, &self.config.history)?;
        self.begun = true;
        self.queue(0, METADATA_TRACK, header_json);
        Ok(())
    }

    /// Time of field `n` (from 0) of the capture, in ns.
    fn field_ns(&self, n: u64) -> i64 {
        let n = i128::from(n);
        let ns = if self.pal { n * 1_000_000_000 / 50 } else { n * 1_000_000_000 * 1001 / 60_000 };
        ns as i64
    }

    fn queue(&mut self, ns: i64, track: u32, data: Vec<u8>) {
        self.queue_frames(ns, track, vec![data]);
    }

    fn queue_frames(&mut self, ns: i64, track: u32, frames: Vec<Vec<u8>>) {
        self.pending.push(Pending { ns, seq: self.seq, track, frames });
        self.seq += 1;
    }

    /// The block of a sample track: the bytes as they are, or the FLAC frames
    /// of the samples (`signed` gives each sample as the encoder takes it).
    fn sample_unit<T: Copy>(
        &self,
        values: &[T],
        channels: usize,
        bytes: impl Fn(T) -> [u8; 2],
        signed: impl Fn(T) -> i32,
    ) -> Result<Vec<Vec<u8>>> {
        Ok(match self.config.codec {
            Codec::Raw => vec![le_bytes(values, bytes)],
            Codec::Flac => {
                let samples: Vec<i32> = values.iter().map(|&v| signed(v)).collect();
                flac::encode(&samples, channels)?
            }
        })
    }

    /// Write the queued blocks up to `ns`, in time order (queue order on a tie,
    /// so a field's metadata goes before its video).
    fn flush_until(&mut self, ns: i64) -> Result<()> {
        self.pending.sort_by_key(|p| (p.ns, p.seq));
        let ready = self.pending.iter().take_while(|p| p.ns <= ns).count();
        let writer = self.writer.as_mut().context("LDX output already finished")?;
        for p in self.pending.drain(..ready) {
            writer.write_frames(p.track, p.ns, p.frames)?;
        }
        Ok(())
    }

    /// One decoded field: its metadata entry, its picture, and its audio and
    /// EFM. `begin` must have been called.
    pub fn write_field(&mut self, field_json: Vec<u8>, luma: &[u16], audio: &[i16], efm: &[u8]) -> Result<()> {
        let ns = self.field_ns(self.fields);
        self.queue(ns, METADATA_TRACK, field_json);
        // FLAC codes signed samples: the unsigned TBC words are made signed by
        // flipping the top bit (subtracting 32768), section 6.1.
        let picture = self.sample_unit(luma, 1, u16::to_le_bytes, |v| i32::from(v) - 32768)?;
        self.queue_frames(ns, TBC_TRACK, picture);
        if let Some(track) = self.pcm_track {
            if !audio.is_empty() {
                // A pcm block is stamped with the time of its first sample.
                let pcm_ns = (i128::from(self.pcm_samples) * 1_000_000_000 / PCM_RATE) as i64;
                let unit = self.sample_unit(audio, 2, i16::to_le_bytes, i32::from)?;
                self.queue_frames(pcm_ns, track, unit);
                self.pcm_samples += (audio.len() / 2) as u64;
            }
        }
        if let Some(track) = self.efm_track {
            self.efm_buf.extend_from_slice(efm);
            while self.efm_buf.len() >= BYTE_CHUNK {
                let chunk: Vec<u8> = self.efm_buf.drain(..BYTE_CHUNK).collect();
                self.queue(ns, track, chunk);
            }
        }
        self.last_ns = ns;
        self.fields += 1;
        self.flush_until(ns)
    }

    /// The rest of the queue, the last EFM chunk, the whole metadata set
    /// (`final_json`, the `.tbc.json` content) and the end of the file.
    pub fn finish(mut self, final_json: Option<Vec<u8>>) -> Result<()> {
        if let Some(track) = self.efm_track {
            if !self.efm_buf.is_empty() {
                let chunk = std::mem::take(&mut self.efm_buf);
                self.queue(self.last_ns, track, chunk);
            }
        }
        self.flush_until(i64::MAX)?;
        let mut writer = self.writer.take().context("LDX output already finished")?;
        if let Some(json) = final_json {
            writer.write_block(METADATA_TRACK, self.last_ns, json)?;
        }
        writer.finish()
    }
}

/// The `LDX_HISTORY` of a decode: one entry, this run.
pub fn history(detail: &str) -> Value {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    json!([{
        "tool": "ld-decode",
        "version": env!("CARGO_PKG_VERSION"),
        "command": "decode",
        "time": utc_now_iso(),
        "detail": detail,
        "arguments": arguments,
    }])
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
fn utc_now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Days since 1970-01-01 to a proleptic Gregorian date (H. Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A `Write` whose bytes stay readable after it is boxed into a stream.
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Read an EBML id at `p`.
    fn read_id(b: &[u8], p: usize) -> (u32, usize) {
        let n = b[p].leading_zeros() as usize + 1;
        let mut v = 0u32;
        for &x in &b[p..p + n] {
            v = (v << 8) | u32::from(x);
        }
        (v, n)
    }

    /// Read an EBML size at `p` (`None` = unknown).
    fn read_size(b: &[u8], p: usize) -> (Option<u64>, usize) {
        let n = b[p].leading_zeros() as usize + 1;
        let mut v = u64::from(b[p] & (0xFFu16 >> n) as u8);
        for &x in &b[p + 1..p + n] {
            v = (v << 8) | u64::from(x);
        }
        let unknown = v == (1u64 << (7 * n)) - 1;
        ((!unknown).then_some(v), n)
    }

    /// The top-level children of the segment: (id, offset in the segment data,
    /// data range).
    fn segment_children(b: &[u8]) -> Vec<(u32, u64, std::ops::Range<usize>)> {
        let (_, n) = read_id(b, 0);
        let (size, m) = read_size(b, n);
        let mut p = n + m + size.unwrap() as usize; // after the EBML header
        let (seg, n) = read_id(b, p);
        assert_eq!(seg, id::SEGMENT);
        let (seg_size, m) = read_size(b, p + n);
        p += n + m;
        let data_start = p;
        let end = seg_size.map_or(b.len(), |s| data_start + s as usize);
        assert_eq!(end, b.len(), "the segment must end where the file does");
        let mut out = Vec::new();
        while p < end {
            let (eid, n) = read_id(b, p);
            let (size, m) = read_size(b, p + n);
            let start = p + n + m;
            let stop = start + size.unwrap() as usize;
            out.push((eid, (p - data_start) as u64, start..stop));
            p = stop;
        }
        out
    }

    #[test]
    fn vint_lengths_and_markers() {
        assert_eq!(vint(0), vec![0x80]);
        assert_eq!(vint(126), vec![0xFE]);
        assert_eq!(vint(127), vec![0x40, 0x7F]); // 0xFF is the reserved "unknown"
        assert_eq!(vint(16_382), vec![0x7F, 0xFE]);
        assert_eq!(vint(16_383), vec![0x20, 0x3F, 0xFF]);
        assert_eq!(id_bytes(id::SEGMENT), vec![0x18, 0x53, 0x80, 0x67]);
        assert_eq!(id_bytes(id::BLOCK), vec![0xA1]);
    }

    #[test]
    fn utc_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_376), (2025, 10, 15));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }

    #[test]
    fn signed_lace_sizes() {
        assert_eq!(signed_vint(0), vec![0xBF]); // 0 + 63, one byte
        assert_eq!(signed_vint(-63), vec![0x80]);
        assert_eq!(signed_vint(63), vec![0xFE]);
        assert_eq!(signed_vint(64), vec![0x60, 0x3F]); // 64 + 8191, two bytes
        assert_eq!(signed_vint(-64), vec![0x5F, 0xBF]);
    }

    fn sample_output(writer: LdxWriter, codec: Codec, luma: &[u16]) -> Result<()> {
        let mut out = LdxOutput::new(
            writer,
            LdxOutputConfig { name: "test".into(), pcm: true, efm: true, codec, history: history("test") },
        );
        let video = VideoGeometry {
            system: "NTSC".into(),
            field_width: 4,
            field_height: luma.len() / 4,
            sample_rate: 14318181.8,
        };
        out.begin(&video, br#"{"videoParameters":{"numberOfSequentialFields":0}}"#.to_vec())?;
        for f in 0..4u16 {
            let field: Vec<u16> = luma.iter().map(|&v| v.wrapping_add(f)).collect();
            out.write_field(
                format!(r#"{{"seqNo":{}}}"#, f + 1).into_bytes(),
                &field,
                &[f as i16; 1470],
                &[3u8; 40_000],
            )?;
        }
        out.finish(Some(br#"{"videoParameters":{},"fields":[]}"#.to_vec()))
    }

    /// The frames of a block's data, undoing its lacing (none or EBML).
    fn unlace(flags: u8, data: &[u8]) -> Vec<Vec<u8>> {
        match (flags >> 1) & 3 {
            0 => vec![data.to_vec()],
            3 => {
                let count = data[0] as usize + 1;
                let mut p = 1;
                let (first, n) = read_size(data, p);
                p += n;
                let mut sizes = vec![first.unwrap() as i64];
                for _ in 1..count - 1 {
                    let (raw, n) = read_size(data, p);
                    p += n;
                    let bias = (1i64 << (7 * n - 1)) - 1;
                    sizes.push(sizes.last().unwrap() + raw.unwrap() as i64 - bias);
                }
                let laced: i64 = sizes.iter().sum();
                sizes.push(data.len() as i64 - p as i64 - laced);
                let mut frames = Vec::new();
                for s in sizes {
                    frames.push(data[p..p + s as usize].to_vec());
                    p += s as usize;
                }
                assert_eq!(p, data.len());
                frames
            }
            other => panic!("lacing {other} is not written"),
        }
    }

    struct BlockSeen {
        us: i64,
        track: u32,
        frames: Vec<Vec<u8>>,
    }

    /// Every block of the clusters, in file order.
    fn blocks_of(b: &[u8]) -> Vec<BlockSeen> {
        let mut blocks = Vec::new();
        for (_, _, range) in segment_children(b).iter().filter(|c| c.0 == id::CLUSTER) {
            let mut p = range.start;
            let (tid, n) = read_id(b, p);
            assert_eq!(tid, id::TIMESTAMP);
            let (size, m) = read_size(b, p + n);
            let ts_start = p + n + m;
            let mut cluster_us = 0u64;
            for &x in &b[ts_start..ts_start + size.unwrap() as usize] {
                cluster_us = (cluster_us << 8) | u64::from(x);
            }
            p = ts_start + size.unwrap() as usize;
            while p < range.end {
                let (gid, n) = read_id(b, p);
                assert_eq!(gid, id::BLOCK_GROUP);
                let (gsize, m) = read_size(b, p + n);
                let (bid, bn) = read_id(b, p + n + m);
                assert_eq!(bid, id::BLOCK);
                let (bsize, bm) = read_size(b, p + n + m + bn);
                let h = p + n + m + bn + bm;
                let track = u32::from(b[h] & 0x7F);
                let rel = i16::from_be_bytes([b[h + 1], b[h + 2]]);
                assert!(rel >= 0 && i64::from(rel) < CLUSTER_SPAN_US);
                let frames = unlace(b[h + 3], &b[h + 4..h + bsize.unwrap() as usize]);
                blocks.push(BlockSeen { us: cluster_us as i64 + i64::from(rel), track, frames });
                p += n + m + gsize.unwrap() as usize;
            }
        }
        blocks
    }

    #[test]
    fn a_file_has_sized_segment_tags_clusters_and_one_cue_per_block() {
        let path = std::env::temp_dir().join(format!("ldx-test-{}.ldx", std::process::id()));
        sample_output(LdxWriter::create(&path).unwrap(), Codec::Raw, &[7u16; 8]).unwrap();
        let b = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let children = segment_children(&b);
        let ids: Vec<u32> = children.iter().map(|c| c.0).collect();
        assert_eq!(&ids[..4], &[id::INFO, id::TRACKS, id::VOID, id::TAGS]);
        assert_eq!(*ids.last().unwrap(), id::CUES);

        let blocks = blocks_of(&b);
        // In time order, all tracks together.
        assert!(blocks.windows(2).all(|w| w[0].us <= w[1].us));
        // 1 parameters + 4 fields + 1 whole set, 4 fields, 4 pcm, 3 efm chunks
        // (160,000 bytes = 2 full + 1 short).
        let count = |t: u32| blocks.iter().filter(|x| x.track == t).count();
        assert_eq!((count(1), count(2), count(3), count(4)), (4, 6, 4, 3));
        // Field times: 0, 16.683, 33.367, 50.050 ms.
        let tbc_us: Vec<i64> = blocks.iter().filter(|x| x.track == 1).map(|x| x.us).collect();
        assert_eq!(tbc_us, vec![0, 16_683, 33_367, 50_050]);
        // Raw: one frame per block, the little-endian samples.
        let first = blocks.iter().find(|x| x.track == 1).unwrap();
        assert_eq!(first.frames, vec![[7u8, 0].repeat(8)]);
        // The EFM chunks put end to end are the EFM.
        let efm: usize = blocks.iter().filter(|x| x.track == 4).map(|x| x.frames[0].len()).sum();
        assert_eq!(efm, 160_000);

        // One cue per block.
        let cues = &children.last().unwrap().2;
        let mut p = cues.start;
        let mut n_cues = 0;
        while p < cues.end {
            let (cid, n) = read_id(&b, p);
            assert_eq!(cid, id::CUE_POINT);
            let (size, m) = read_size(&b, p + n);
            p += n + m + size.unwrap() as usize;
            n_cues += 1;
        }
        assert_eq!(n_cues, blocks.len());
    }

    #[test]
    fn a_stream_has_no_size_no_void_and_no_cues() {
        let shared = Shared::default();
        sample_output(LdxWriter::stream(Box::new(shared.clone())), Codec::Raw, &[0u16; 8]).unwrap();
        let b = shared.0.lock().unwrap().clone();
        let ids: Vec<u32> = segment_children(&b).iter().map(|c| c.0).collect();
        assert_eq!(&ids[..3], &[id::INFO, id::TRACKS, id::TAGS]);
        assert!(!ids.contains(&id::VOID));
        assert!(!ids.contains(&id::CUES));
        assert!(ids.contains(&id::CLUSTER));
    }

    #[test]
    fn tags_carry_the_profile_and_each_track() {
        let shared = Shared::default();
        sample_output(LdxWriter::stream(Box::new(shared.clone())), Codec::Raw, &[0u16; 8]).unwrap();
        let b = shared.0.lock().unwrap().clone();
        let text = String::from_utf8_lossy(&b);
        assert!(text.contains("LDX_PROFILE"));
        assert!(text.contains("LDX_NEXT_TRACK_ID"));
        assert!(text.contains(r#""kind":"tbc""#));
        assert!(text.contains(r#""codec":"raw""#));
        assert!(text.contains(r#""namespace":"ldx.tbcmetadata""#));
        assert!(text.contains("D_LDX/EFM"));
        assert!(text.contains("A_PCM/INT/LIT"));
    }

    /// Decode the frames of one FLAC unit with claxon, given the track's
    /// `CodecPrivate`: what a reader of the file gets back.
    fn decode_flac(private: &[u8], frames: &[Vec<u8>]) -> Vec<i32> {
        let mut stream = private.to_vec();
        for f in frames {
            stream.extend_from_slice(f);
        }
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(stream)).unwrap();
        let mut out = Vec::new();
        let mut buffer = Vec::new();
        let mut blocks = reader.blocks();
        while let Some(block) = blocks.read_next_or_eof(std::mem::take(&mut buffer)).unwrap() {
            for i in 0..block.duration() {
                for ch in 0..block.channels() {
                    out.push(block.sample(ch, i));
                }
            }
            buffer = block.into_buffer();
        }
        out
    }

    #[test]
    fn flac_tracks_give_the_samples_back() {
        // A field of a full NTSC TBC: 239,330 samples = 7 frames of 32,768 and
        // one shorter, laced in one block.
        let luma: Vec<u16> = (0..910 * 263u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 16) as u16).collect();
        let shared = Shared::default();
        sample_output(LdxWriter::stream(Box::new(shared.clone())), Codec::Flac, &luma).unwrap();
        let b = shared.0.lock().unwrap().clone();
        let text = String::from_utf8_lossy(&b);
        assert!(text.contains("A_FLAC"));
        assert!(text.contains(r#""codec":"flac""#));

        let blocks = blocks_of(&b);
        let tbc: Vec<&BlockSeen> = blocks.iter().filter(|x| x.track == 1).collect();
        assert_eq!(tbc.len(), 4);
        for (f, block) in tbc.iter().enumerate() {
            assert_eq!(block.frames.len(), 8, "7 full frames and a short one");
            // The unsigned TBC words come back with their top bit flipped.
            let got = decode_flac(&flac::codec_private(1).unwrap(), &block.frames);
            let want: Vec<i32> = luma.iter().map(|&v| i32::from(v.wrapping_add(f as u16)) - 32768).collect();
            assert!(got == want, "field {f} does not decode to its samples");
        }
        // The audio: stereo, one frame per field.
        let pcm = blocks.iter().find(|x| x.track == 3).unwrap();
        assert_eq!(pcm.frames.len(), 1);
        assert_eq!(decode_flac(&flac::codec_private(2).unwrap(), &pcm.frames), vec![0i32; 1470]);
        // The metadata and EFM tracks stay raw.
        assert!(blocks.iter().filter(|x| x.track == 2 || x.track == 4).all(|x| x.frames.len() == 1));
    }

    #[test]
    fn ffv1_is_refused_for_a_tbc_track() {
        assert_eq!(CodecChoice::Raw.for_tbc().unwrap(), Codec::Raw);
        assert_eq!(CodecChoice::Flac.for_tbc().unwrap(), Codec::Flac);
        assert_eq!(CodecChoice::Auto.for_tbc().unwrap(), Codec::Flac);
        assert!(CodecChoice::Ffv1.for_tbc().is_err());
    }
}
