/// Errors from ranga image processing operations.
#[derive(Debug, thiserror::Error)]
pub enum RangaError {
    #[error("invalid pixel format: {0}")]
    InvalidFormat(String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("buffer too small: need {need} bytes, have {have}")]
    BufferTooSmall { need: usize, have: usize },

    #[error("unsupported conversion: {from} → {to}")]
    UnsupportedConversion { from: String, to: String },

    #[error("{0}")]
    Other(String),
}
