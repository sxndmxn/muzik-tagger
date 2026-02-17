use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::db;
use crate::models::{METADATA_DIM, Track, is_zero_vec};

use super::{ensure_models_dir, normalize};

const MODEL_NAME: &str = "all-MiniLM-L6-v2";

/// Download the `MiniLM` ONNX model and tokenizer if not present.
fn ensure_model() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = ensure_models_dir()?.join(MODEL_NAME);
    let model_path = dir.join("model.onnx");
    let tokenizer_path = dir.join("tokenizer.json");

    if model_path.exists() && tokenizer_path.exists() {
        return Ok((model_path, tokenizer_path));
    }

    std::fs::create_dir_all(&dir)?;

    let base_url = format!(
        "https://huggingface.co/sentence-transformers/{MODEL_NAME}/resolve/main"
    );

    if !model_path.exists() {
        eprintln!("Downloading {MODEL_NAME} ONNX model...");
        let url = format!("{base_url}/onnx/model.onnx");
        download_file(&url, &model_path)?;
    }

    if !tokenizer_path.exists() {
        eprintln!("Downloading {MODEL_NAME} tokenizer...");
        let url = format!("{base_url}/tokenizer.json");
        download_file(&url, &tokenizer_path)?;
    }

    Ok((model_path, tokenizer_path))
}

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    let output = std::process::Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(dest)
        .arg(url)
        .output()
        .context("failed to run curl")?;

    if !output.status.success() {
        bail!(
            "Failed to download {}: {}",
            url,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

/// Mean pooling with attention mask.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn mean_pooling(
    token_embeddings: &[f32],
    shape: &[i64],
    attention_mask: &[i64],
) -> Vec<f32> {
    // shape is [batch, seq_len, hidden_dim]
    let seq_len = shape[1] as usize;
    let dim = shape[2] as usize;
    let mut result = vec![0.0f32; dim];
    let mut total_weight = 0.0f32;

    for (i, &mask) in attention_mask.iter().enumerate().take(seq_len) {
        if mask > 0 {
            #[allow(clippy::cast_precision_loss)]
            let weight = mask as f32;
            total_weight += weight;
            let offset = i * dim;
            for j in 0..dim {
                result[j] += token_embeddings[offset + j] * weight;
            }
        }
    }

    if total_weight > 0.0 {
        for v in &mut result {
            *v /= total_weight;
        }
    }

    result
}

/// Generate metadata text embeddings for all tracks.
///
/// Returns the count of embedded tracks.
pub fn embed_metadata(quiet: bool) -> Result<usize> {
    let mut db = db::open_db()?;
    let tracks = db::all_tracks(&db)?;

    if tracks.is_empty() {
        bail!("No tracks in database. Run 'mzk scan' first.");
    }

    let (model_path, tokenizer_path) = ensure_model()?;
    let mut session = Session::builder()?
        .commit_from_file(&model_path)
        .context("failed to load MiniLM ONNX model")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let texts: Vec<String> = tracks.iter().map(Track::metadata_text).collect();

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(u64::try_from(tracks.len()).unwrap_or(u64::MAX));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} Embedding metadata")
                .map_err(|e| anyhow::anyhow!("progress template: {e}"))?,
        );
        pb
    };

    let mut updated_tracks = tracks;

    for (idx, text) in texts.iter().enumerate() {
        pb.inc(1);

        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| i64::from(m)).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| i64::from(t)).collect();
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask.clone()))?;
        let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let mut vec = mean_pooling(data, shape, &attention_mask);
        vec.truncate(METADATA_DIM);
        normalize(&mut vec);

        updated_tracks[idx].metadata_vec = vec;
    }

    db::write_all_tracks(&mut db, &updated_tracks)?;

    pb.finish_and_clear();

    // Aggregate album metadata vectors
    aggregate_album_metadata(&mut db)?;

    Ok(updated_tracks.len())
}

/// Compute album-level metadata vectors as mean of track vectors.
fn aggregate_album_metadata(db: &mut db::Db) -> Result<()> {
    let tracks = db::all_tracks(db)?;
    let mut albums = db::all_albums(db)?;

    let mut album_vecs: std::collections::HashMap<String, Vec<Vec<f32>>> =
        std::collections::HashMap::new();

    for track in &tracks {
        if is_zero_vec(&track.metadata_vec) {
            continue;
        }
        let key = format!(
            "{}||{}",
            track.album_artist.to_lowercase().trim(),
            track.album.to_lowercase().trim()
        );
        album_vecs
            .entry(key)
            .or_default()
            .push(track.metadata_vec.clone());
    }

    let mut updated = Vec::new();
    for album in &mut albums {
        let key = format!(
            "{}||{}",
            album.artist.to_lowercase().trim(),
            album.album.to_lowercase().trim()
        );
        if let Some(vecs) = album_vecs.get(&key) {
            let dim = METADATA_DIM;
            let mut mean = vec![0.0f32; dim];
            for vec in vecs {
                for (i, &v) in vec.iter().enumerate() {
                    mean[i] += v;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let n = vecs.len() as f32;
            for v in &mut mean {
                *v /= n;
            }
            normalize(&mut mean);
            album.metadata_vec = mean;
            updated.push(album.clone());
        }
    }

    if !updated.is_empty() {
        db::write_all_albums(db, &albums)?;
    }

    Ok(())
}
