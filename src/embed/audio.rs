use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use ort::session::Session;
use ort::value::Tensor;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::db;
use crate::models::{AUDIO_DIM, is_zero_vec};

use super::{ensure_models_dir, normalize};

// ─── Constants ─────────────────────────────────────────────────────────

const CLAP_MODEL_NAME: &str = "clap-htsat-unfused";
const CLAP_BASE_URL: &str = "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/main";
const SAMPLE_RATE: u32 = 48000;
const TARGET_SAMPLES: usize = SAMPLE_RATE as usize * 10; // 10 seconds

// Mel spectrogram parameters (match HuggingFace `ClapFeatureExtractor`)
const N_FFT: usize = 1024;
const HOP_LENGTH: usize = 480;
const N_MELS: usize = 64;
const FREQ_MIN: f32 = 0.0;
const FREQ_MAX: f32 = 14_000.0;

// ─── Mel Spectrogram ───────────────────────────────────────────────────

/// Periodic Hann window (matches `np.hanning(n+1)[:-1]`).
#[allow(clippy::cast_precision_loss)]
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / size as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// Hz to mel (HTK scale).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Mel to Hz (HTK scale).
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular mel filterbank: `N_MELS` filters of length `N_FFT / 2 + 1`.
///
/// HTK mel scale, no area normalization
/// (matches `HuggingFace` `norm=None, mel_scale="htk"`).
#[allow(clippy::cast_precision_loss)]
fn mel_filterbank() -> Vec<Vec<f32>> {
    let freq_bins = N_FFT / 2 + 1;
    let mel_lo = hz_to_mel(FREQ_MIN);
    let mel_hi = hz_to_mel(FREQ_MAX);

    let mel_points: Vec<f32> = (0..=N_MELS + 1)
        .map(|i| mel_lo + (mel_hi - mel_lo) * i as f32 / (N_MELS + 1) as f32)
        .collect();

    let bin_freqs: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) * (N_FFT as f32 + 1.0) / SAMPLE_RATE as f32)
        .collect();

    let mut filters = vec![vec![0.0f32; freq_bins]; N_MELS];

    for (i, row) in filters.iter_mut().enumerate() {
        let left = bin_freqs[i];
        let center = bin_freqs[i + 1];
        let right = bin_freqs[i + 2];
        let up = center - left;
        let down = right - center;

        for (j, val) in row.iter_mut().enumerate() {
            let freq = j as f32;
            if freq > left && freq <= center && up > 0.0 {
                *val = (freq - left) / up;
            } else if freq > center && freq < right && down > 0.0 {
                *val = (right - freq) / down;
            }
        }
    }

    filters
}

/// Reflect-pad signal (matches `np.pad(signal, pad, mode='reflect')`).
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut result = Vec::with_capacity(n + 2 * pad);

    for i in (1..=pad).rev() {
        result.push(signal[i.min(n - 1)]);
    }
    result.extend_from_slice(signal);
    for i in 1..=pad {
        result.push(signal[(n - 1).saturating_sub(i)]);
    }

    result
}

/// STFT power spectrogram with center reflection padding.
///
/// Returns `num_frames` rows of `N_FFT / 2 + 1` power values.
fn compute_stft(signal: &[f32], window: &[f32]) -> Vec<Vec<f32>> {
    let padded = reflect_pad(signal, N_FFT / 2);
    let freq_bins = N_FFT / 2 + 1;
    let num_frames = if padded.len() >= N_FFT {
        (padded.len() - N_FFT) / HOP_LENGTH + 1
    } else {
        return vec![];
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buffer = vec![Complex::new(0.0f32, 0.0); N_FFT];
    let mut frames = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * HOP_LENGTH;
        for (k, slot) in buffer.iter_mut().enumerate() {
            *slot = Complex::new(padded[start + k] * window[k], 0.0);
        }
        fft.process(&mut buffer);
        let power: Vec<f32> = buffer[..freq_bins]
            .iter()
            .map(rustfft::num_complex::Complex::norm_sqr)
            .collect();
        frames.push(power);
    }

    frames
}

/// Full mel spectrogram: PCM → STFT → mel filterbank → log dB.
///
/// Returns `(flat_data, num_frames)` where `flat_data` is row-major
/// `[num_frames × N_MELS]` for ONNX tensor construction.
fn compute_mel_spectrogram(pcm: &[f32]) -> (Vec<f32>, usize) {
    let window = hann_window(N_FFT);
    let power_spec = compute_stft(pcm, &window);
    let filters = mel_filterbank();
    let num_frames = power_spec.len();

    let mut flat = Vec::with_capacity(num_frames * N_MELS);

    for frame in &power_spec {
        for filter in &filters {
            let energy: f32 = frame
                .iter()
                .zip(filter.iter())
                .map(|(&p, &f)| p * f)
                .sum();
            // Power to dB: 10 * log10(max(energy, 1e-10))
            flat.push(10.0 * energy.max(1e-10).log10());
        }
    }

    (flat, num_frames)
}

// ─── File download ─────────────────────────────────────────────────────

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    let status = std::process::Command::new("curl")
        .args(["-fSL", "--progress-bar", "-o"])
        .arg(&tmp)
        .arg(url)
        .status()
        .context("failed to run curl")?;

    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!("Failed to download {url}");
    }

    std::fs::rename(&tmp, dest)?;
    Ok(())
}

// ─── Model management ──────────────────────────────────────────────────

/// Download the CLAP audio ONNX model if not present.
fn ensure_clap_model() -> Result<std::path::PathBuf> {
    let dir = ensure_models_dir()?.join(CLAP_MODEL_NAME);
    let model_path = dir.join("audio_model.onnx");

    if model_path.exists() {
        return Ok(model_path);
    }

    std::fs::create_dir_all(&dir)?;

    eprintln!("Downloading CLAP audio model (118 MB)...");
    let url = format!("{CLAP_BASE_URL}/onnx/audio_model.onnx");
    download_file(&url, &model_path)?;
    eprintln!("Saved to {}", model_path.display());

    Ok(model_path)
}

/// Download the CLAP text ONNX model and tokenizer if not present.
fn ensure_clap_text_model() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = ensure_models_dir()?.join(CLAP_MODEL_NAME);
    let model_path = dir.join("text_model.onnx");
    let tokenizer_path = dir.join("tokenizer.json");

    if model_path.exists() && tokenizer_path.exists() {
        return Ok((model_path, tokenizer_path));
    }

    std::fs::create_dir_all(&dir)?;

    if !model_path.exists() {
        eprintln!("Downloading CLAP text model (502 MB)...");
        let url = format!("{CLAP_BASE_URL}/onnx/text_model.onnx");
        download_file(&url, &model_path)?;
    }

    if !tokenizer_path.exists() {
        eprintln!("Downloading CLAP tokenizer...");
        let url = format!("{CLAP_BASE_URL}/tokenizer.json");
        download_file(&url, &tokenizer_path)?;
    }

    Ok((model_path, tokenizer_path))
}

// ─── Audio decoding ────────────────────────────────────────────────────

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
        if num_frames == 0 {
            continue;
        }
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

// ─── Embedding ─────────────────────────────────────────────────────────

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
                .map_err(|e| anyhow::anyhow!("progress template: {e}"))?,
        );
        pb
    };

    let mut updated_tracks = tracks;
    let mut count = 0usize;

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

        // Compute mel spectrogram for CLAP input
        let (mel_data, num_frames) = compute_mel_spectrogram(&pcm);
        let mel_tensor = match Tensor::from_array(([1usize, 1, num_frames, N_MELS], mel_data)) {
            Ok(t) => t,
            Err(e) => {
                if !quiet {
                    eprintln!("  skip {filepath}: mel tensor error: {e}");
                }
                continue;
            }
        };

        let outputs = match session.run(ort::inputs!["input_features" => mel_tensor]) {
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
        }
    }

    pb.finish_and_clear();

    if count > 0 {
        db::write_all_tracks(&mut db, &updated_tracks)?;
    }

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
        db::write_all_albums(db, &albums)?;
    }

    Ok(())
}

/// Encode a text query into CLAP's audio-text shared space.
pub fn clap_text_embed(query: &str) -> Result<Vec<f32>> {
    let (model_path, tokenizer_path) = ensure_clap_text_model()?;
    let mut session = Session::builder()?
        .commit_from_file(&model_path)
        .context("failed to load CLAP text ONNX model")?;

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load CLAP tokenizer: {e}"))?;

    let encoding = tokenizer
        .encode(query, true)
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
    let seq_len = input_ids.len();

    let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))?;

    let outputs = session.run(ort::inputs!["input_ids" => input_ids_tensor])?;

    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let mut vec: Vec<f32> = data.to_vec();
    vec.truncate(AUDIO_DIM);
    normalize(&mut vec);

    Ok(vec)
}
