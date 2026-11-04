//! `rfx`: reading the `.rfx` archives, several RF captures of one disc in a
//! single lossless file. The archives are written by the `rfx` tool of
//! RF-tools; the format is specified there.
pub mod align;
pub mod archive;
pub mod ebml;
pub mod flacmode;
pub mod fmt;
pub mod gpu;
pub mod io;
pub mod model;
pub mod pred;
pub mod rc;
pub mod reader;
pub mod seg;
