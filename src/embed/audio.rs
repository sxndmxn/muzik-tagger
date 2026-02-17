use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use ort::session::Session;
use ort::value::Tensor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::db;
use crate::models::{AUDIO_DIM, is_zero_vec};

use super::{ensure_models_dir, normalize};

const CLAP_MODEL_NAME: &str = "clap-htsat-tiny";
const DB_BATCH_SIZE: usize = 256;
const SAMPLE_RATE: u32 = 48000;
const TARGET_SAMPLES: usize = SAMPLE_RATE as usize * 10; // 10 seconds

/// Download the CLAP ONNX model if not present.
fn ensure_clap_model() -> Result<std::path::PathBuf> {
    let dir = ensure_models_dir()?.join(CLAP_MODEL_NAME);
    let model_path = dir.join("audio_model.onnx");

    if model_path.exists() {
        return Ok(model_path);
    }

    std::fs::create_dir_all(&dir)?;

    eprintln!("CLAP audio ONNX model not found.");
    eprintln!("Place the audio model at: {}", model_path.display());
    eprintln!("Export from: https://huggingface.co/laion/clap-htsat-unfused");
    bail!(
        "CLAP ONNX model not found at {}",
        model_path.display()
    );
}

/// Ensure the CLAP text encoder model is available.
fn ensure_clap_text_model() -> Result<std::path::PathBuf> {
    let dir = ensure_models_dir()?.join(CLAP_MODEL_NAME);
    let model_path = dir.join("text_model.onnx");

    if model_path.exists() {
        return Ok(model_path);
    }

    std::fs::create_dir_all(&dir)?;

    eprintln!("CLAP text model not found at: {}", model_path.display());
    eprintln!("Place the text encoder ONNX model there for NL search.");
    bail!(
        "CLAP text ONNX model not found at {}",
        model_path.display()
    );
}

/// Decode audio file to mono PCM f32 samples at 48kHz.
#[allow(clippy::similar_names)]
fn decode_audio(filepath: &str) -> Result<Vec<f32>> {
    use symphonia::core::io::MediaSourceStreamOptions;

    let file = std::fs::File::open(filepath)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(filepath).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .context("no audio track found")?;

    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())?;

    let mut samples = Vec::new();
    let source_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params
        .channels
        .map_or(1, symphonia::core::audio::Channels::count);

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let Ok(decoded) = decoder.decode(&packet) else {
            continue;
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();
        let mut sample_buf = SampleBuffer::<f32>::new(
            u64::try_from(num_frames).unwrap_or(u64::MAX),
            spec,
        );
        sample_buf.copy_interleaved_ref(decoded);

        let interleaved = sample_buf.samples();

        // Mix down to mono
        for frame in interleaved.chunks(channels) {
            #[allow(clippy::cast_precision_loss)]
            let mono: f32 = frame.iter().sum::<f32>() / channels as f32;
            samples.push(mono);
        }

        // Stop after enough samples
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
        let target = (TARGET_SAMPLES as f64 * f64::from(source_rate) / f64::from(SAMPLE_RATE)) as usize;
        if samples.len() >= target {
            break;
        }
    }

    // Simple resample to 48kHz if needed
    if source_rate != SAMPLE_RATE {
        samples = resample(&samples, source_rate, SAMPLE_RATE);
    }

    // Pad or truncate to target length
    samples.resize(TARGET_SAMPLES, 0.0);

    Ok(samples)
}

/// Simple linear resampling.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = f64::from(from_rate) / f64::from(to_rate);
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx as usize;
        let frac = (src_idx - idx0 as f64) as f32;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        output.push(input[idx0] * (1.0 - frac) + input[idx1] * frac);
    }

    output
}

/// Generate CLAP audio embeddings for all tracks.
///
/// Returns the count of newly embedded tracks.
pub fn embed_audio(quiet: bool) -> Result<usize> {
    let mut db = db::open_db()?;
    let tracks = db::all_tracks(&db)?;

    if tracks.is_empty() {
        bail!("No tracks in database. Run 'mzk scan' first.");
    }

    // Filter to tracks needing audio embeddings
    let indices: Vec<usize> = tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| is_zero_vec(&t.audio_vec))
        .map(|(i, _)| i)
        .collect();

    if indices.is_empty() {
        return Ok(0);
    }

    let model_path = ensure_clap_model()?;
    let mut session = Session::builder()?
        .commit_from_file(&model_path)
        .context("failed to load CLAP ONNX model")?;

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(u64::try_from(indices.len()).unwrap_or(u64::MAX));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} Embedding audio (CLAP)")
                .expect("template"),
        );
        pb
    };

    let mut updated_tracks = tracks;
    let mut count = 0usize;
    let mut batch_updates: Vec<usize> = Vec::new();

    for &idx in &indices {
        pb.inc(1);
        let filepath = &updated_tracks[idx].filepath;

        let pcm = match decode_audio(filepath) {
            Ok(p) => p,
            Err(e) => {
                if !quiet {
                    eprintln!("  skip {filepath}: {e}");
                }
                continue;
            }
        };

        // Run through CLAP model
        let input_tensor = match Tensor::from_array(([1usize, pcm.len()], pcm)) {
            Ok(t) => t,
            Err(e) => {
                if !quiet {
                    eprintln!("  skip {filepath}: tensor error: {e}");
                }
                continue;
            }
        };

        let outputs = match session.run(ort::inputs!["input" => input_tensor]) {
            Ok(o) => o,
            Err(e) => {
                if !quiet {
                    eprintln!("  skip {filepath}: inference error: {e}");
                }
                continue;
            }
        };

        let (_shape, data) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(r) => r,
            Err(e) => {
                if !quiet {
                    eprintln!("  skip {filepath}: extract error: {e}");
                }
                continue;
            }
        };

        let mut vec: Vec<f32> = data.to_vec();
        vec.truncate(AUDIO_DIM);
        normalize(&mut vec);

        if vec.len() == AUDIO_DIM {
            updated_tracks[idx].audio_vec = vec;
            count += 1;
            batch_updates.push(idx);
        }

        if batch_updates.len() >= DB_BATCH_SIZE {
            let batch: Vec<_> = batch_updates
                .iter()
                .map(|&i| updated_tracks[i].clone())
                .collect();
            db::update_tracks(&mut db, &batch)?;
            batch_updates.clear();
        }
    }

    if !batch_updates.is_empty() {
        let batch: Vec<_> = batch_updates
            .iter()
            .map(|&i| updated_tracks[i].clone())
            .collect();
        db::update_tracks(&mut db, &batch)?;
    }

    pb.finish_and_clear();

    // Aggregate album audio vectors
    aggregate_album_audio(&mut db)?;

    Ok(count)
}

/// Compute album-level audio vectors as mean of track audio vectors.
fn aggregate_album_audio(db: &mut db::Db) -> Result<()> {
    let tracks = db::all_tracks(db)?;
    let mut albums = db::all_albums(db)?;

    let mut album_vecs: std::collections::HashMap<String, Vec<Vec<f32>>> =
        std::collections::HashMap::new();

    for track in &tracks {
        if is_zero_vec(&track.audio_vec) {
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
            .push(track.audio_vec.clone());
    }

    let mut updated = Vec::new();
    for album in &mut albums {
        let key = format!(
            "{}||{}",
            album.artist.to_lowercase().trim(),
            album.album.to_lowercase().trim()
        );
        if let Some(vecs) = album_vecs.get(&key) {
            let dim = AUDIO_DIM;
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
            album.audio_vec = mean;
            updated.push(album.clone());
        }
    }

    if !updated.is_empty() {
        for chunk in updated.chunks(DB_BATCH_SIZE) {
            db::update_albums(db, chunk)?;
        }
    }

    Ok(())
}

/// Encode a text query into CLAP's audio-text shared space.
pub fn clap_text_embed(query: &str) -> Result<Vec<f32>> {
    let model_path = ensure_clap_text_model()?;
    let mut session = Session::builder()?
        .commit_from_file(&model_path)
        .context("failed to load CLAP text ONNX model")?;

    let tokenizer_path = ensure_models_dir()?.join(CLAP_MODEL_NAME).join("tokenizer.json");

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load CLAP tokenizer: {e}"))?;

    let encoding = tokenizer
        .encode(query, true)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| i64::from(m)).collect();
    let seq_len = input_ids.len();

    let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;
    let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask))?;

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    ])?;

    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let mut vec: Vec<f32> = data.to_vec();
    vec.truncate(AUDIO_DIM);
    normalize(&mut vec);

    Ok(vec)
}
