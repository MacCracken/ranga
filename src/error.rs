/// Errors from ranga image processing operations.
///
/// # Examples
///
/// ```
/// use ranga::RangaError;
///
/// let err = RangaError::BufferTooSmall { need: 1024, have: 512 };
/// assert!(err.to_string().contains("buffer too small"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum RangaError {
    /// The pixel format is invalid or unsupported for this operation.
    #[error("invalid pixel format: {0}")]
    InvalidFormat(String),

    /// Buffer dimensions do not match the expected size.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// The provided buffer is too small for the operation.
    #[error("buffer too small: need {need} bytes, have {have}")]
    BufferTooSmall { need: usize, have: usize },

    /// The requested conversion between formats is not supported.
    #[error("unsupported conversion: {from} → {to}")]
    UnsupportedConversion { from: String, to: String },

    /// A generic error with a descriptive message.
    #[error("{0}")]
    Other(String),
}
