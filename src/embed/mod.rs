pub mod audio;
pub mod metadata;

use std::path::PathBuf;

use anyhow::Result;

/// Directory where ONNX models are stored.
pub fn models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mzk")
        .join("models")
}

/// Ensure the models directory exists.
pub fn ensure_models_dir() -> Result<PathBuf> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// L2-normalize a vector in place.
pub fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut *vec {
            *v /= norm;
        }
    }
}
