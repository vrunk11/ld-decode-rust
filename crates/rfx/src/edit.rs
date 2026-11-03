//! Editing archives: merge several into one, extract some of the tracks into
//! another, remove tracks, add captures. The output is always a new file.
//!
//! A track is coded against the tracks before it, so its coded blocks stay
//! valid exactly when every track it was coded against is still there, before
//! it. Those are copied as they are, which costs no more than the copy (a
//! merge, or the removal of the last tracks, never decodes anything). The
//! others, and the new captures, are coded again against what is left: the
//! archives involved are decoded to a work directory for that, and the usual
//! `prepare` finds the alignment and the references.
use crate::pack::{self, open, pack_prepared, prepare_with, write_head, Meta, Opened, PackOpts, Spec};
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

/// One track of the archive to write.
#[derive(Clone, Debug)]
pub enum Pick {
    /// Track `src` of the archive number `arch` in the list given.
    Old { arch: usize, src: usize },
    /// A capture file.
    New(PathBuf),
}

/// Write `out` from `picks` (in that order). `recode_from`: the archives from
/// that one on are coded again against everything before them, so that
/// captures of one disc in different archives predict each other.
pub fn assemble(archives: &[PathBuf], picks: &[Pick], out: &Path, o: &PackOpts, recode_from: Option<usize>) -> Result<()> {
    if picks.is_empty() {
        bail!("nothing to write: no track selected");
    }
    let opened: Vec<Opened> = archives.iter().map(|p| open(p)).collect::<Result<_>>()?;
    if let Some(a) = opened.first() {
        if opened.iter().any(|b| b.meta.segment != a.meta.segment) {
            bail!("the archives do not use the same segment length");
        }
    }
    let segment = opened.first().map_or(o.segment, |a| a.meta.segment);
    let o = &PackOpts { order: pack::OrderMode::Given, device: o.device, level: o.level, segment, verify: o.verify, pass0_dir: o.pass0_dir.clone(), ld_decode: o.ld_decode.clone(), rate_mhz: o.rate_mhz, format: "auto".into() };

    // Where each old source goes.
    let mut place: HashMap<(usize, usize), usize> = HashMap::new();
    for (k, p) in picks.iter().enumerate() {
        if let Pick::Old { arch, src } = p {
            if *arch >= opened.len() || *src >= opened[*arch].meta.sources.len() {
                bail!("no track {src} in archive {arch}");
            }
            if place.insert((*arch, *src), k).is_some() {
                bail!("a track is selected twice");
            }
        }
    }
    // The spec of every pick, with the coding kept when it stays valid.
    let mut specs: Vec<Spec> = Vec::new();
    let mut known: HashMap<(usize, usize), crate::align::LagTable> = HashMap::new();
    let mut copy: Vec<Option<(&Opened, usize)>> = Vec::new();
    for (k, p) in picks.iter().enumerate() {
        match p {
            Pick::New(_) => {
                specs.push(Spec::default());
                copy.push(None);
            }
            Pick::Old { arch, src } => {
                let a = &opened[*arch];
                let sm = &a.meta.sources[*src];
                let moved: Option<Vec<usize>> = sm.refs.iter().map(|j| place.get(&(*arch, *j)).copied().filter(|&n| n < k)).collect();
                let forced = recode_from.is_some_and(|f| *arch >= f);
                match moved {
                    Some(refs) if !forced => {
                        let mut m = sm.clone();
                        m.refs = refs.clone();
                        for (j_old, j_new) in sm.refs.iter().zip(&refs) {
                            let t = a.tables.get(&(*src, *j_old)).with_context(|| format!("{}: lag table against track {j_old} is missing", sm.name))?;
                            known.insert((k, *j_new), t.clone());
                        }
                        specs.push(Spec { carry: Some(m), frozen: true });
                        copy.push(Some((a, *src)));
                    }
                    _ => {
                        specs.push(Spec { carry: Some(sm.clone()), frozen: false });
                        copy.push(None);
                    }
                }
            }
        }
    }
    let kept = copy.iter().filter(|c| c.is_some()).count();
    eprintln!("{} track(s) to write: {} copied as they are, {} to code", picks.len(), kept, picks.len() - kept);

    let tmp = out.with_extension("rfx-tmp");
    let result = (|| -> Result<()> {
        if kept == picks.len() {
            // Nothing to code: the blocks and the tables, renumbered.
            let metas: Vec<_> = specs.iter().enumerate().map(|(k, s)| pack::SrcMeta { position: k, ..s.carry.clone().expect("old source") }).collect();
            let mut names = HashSet::new();
            if let Some(n) = metas.iter().find(|m| !names.insert(m.name.clone())) {
                bail!("two tracks are named {}: the archive keeps file names only", n.name);
            }
            let meta = Meta { format: "rfx".into(), version: 1, segment, sources: metas };
            let mut w = std::io::BufWriter::new(File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?);
            write_head(&mut w, &meta, &known)?;
            for (k, c) in copy.iter().enumerate() {
                let (a, src) = c.expect("all copied");
                pack::copy_blocks(&mut w, a, src, k)?;
            }
            std::io::Write::flush(&mut w)?;
            return Ok(());
        }
        // Coding needed: the archives' sources become raw files to read the
        // references from.
        let work = out.with_extension("edit");
        std::fs::create_dir_all(&work)?;
        let cleanup = scopeguard(work.clone());
        let mut dirs: HashMap<usize, PathBuf> = HashMap::new();
        for p in picks {
            if let Pick::Old { arch, .. } = p {
                if !dirs.contains_key(arch) {
                    let d = work.join(format!("a{arch}"));
                    eprintln!("decoding {} to read the references", archives[*arch].display());
                    pack::unpack(&archives[*arch], &d, None, o.device, None)?;
                    dirs.insert(*arch, d);
                }
            }
        }
        let given: Vec<PathBuf> = picks
            .iter()
            .map(|p| match p {
                Pick::New(path) => path.clone(),
                Pick::Old { arch, src } => dirs[arch].join(&opened[*arch].meta.sources[*src].name),
            })
            .collect();
        let prepared = prepare_with(&given, &specs, &known, &tmp, o)?;
        pack_prepared(&prepared, &tmp, o, &copy)?;
        drop(prepared);
        drop(cleanup);
        Ok(())
    })();
    drop(copy);
    drop(opened);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if out.exists() {
        std::fs::remove_file(out)?;
    }
    std::fs::rename(&tmp, out).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

/// Removes a directory when dropped.
struct RemoveDir(PathBuf);

impl Drop for RemoveDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scopeguard(p: PathBuf) -> RemoveDir {
    RemoveDir(p)
}

/// The tracks of `archive` listed in `which` (indices, `0` is the primary),
/// as picks in their original order.
pub fn pick_some(archive: &Path, which: &[usize]) -> Result<Vec<Pick>> {
    let n = open(archive)?.meta.sources.len();
    if let Some(bad) = which.iter().find(|&&i| i >= n) {
        bail!("{} has {} tracks: no track {bad}", archive.display(), n);
    }
    let set: HashSet<usize> = which.iter().copied().collect();
    Ok((0..n).filter(|i| set.contains(i)).map(|src| Pick::Old { arch: 0, src }).collect())
}

/// All the tracks of `archive` except `which`.
pub fn pick_without(archive: &Path, which: &[usize]) -> Result<Vec<Pick>> {
    let n = open(archive)?.meta.sources.len();
    if let Some(bad) = which.iter().find(|&&i| i >= n) {
        bail!("{} has {} tracks: no track {bad}", archive.display(), n);
    }
    let set: HashSet<usize> = which.iter().copied().collect();
    Ok((0..n).filter(|i| !set.contains(i)).map(|src| Pick::Old { arch: 0, src }).collect())
}

/// Every track of every archive, one archive after the other.
pub fn pick_all(archives: &[PathBuf]) -> Result<Vec<Pick>> {
    let mut picks = Vec::new();
    for (arch, p) in archives.iter().enumerate() {
        for src in 0..open(p)?.meta.sources.len() {
            picks.push(Pick::Old { arch, src });
        }
    }
    Ok(picks)
}
