// The selectivity threshold at which we switch from random seeks to scan.
pub const SELECTIVITY_THRESHOLD: f32 = 0.2;

pub const SMOOTHING_PARAMETER: f64 = 1e-9;

pub const BLOCK_SIZE: usize = 512;

pub const READ_BUF_SIZE: usize = 8 * 4096;
//pub const READ_BUF_SIZE: usize = 32;
