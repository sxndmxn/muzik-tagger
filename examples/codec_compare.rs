//! Compare PCM decoding and fingerprints across codec versions of the same audio.
//!
//! Usage: cargo run --example codec_compare

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const SAMPLE_RATE: u32 = 48000;
const TARGET_SAMPLES: usize = SAMPLE_RATE as usize * 10;

fn decode_to_pcm(filepath: &str) -> Vec<f32> {
    let file = std::fs::File::open(filepath).unwrap();
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(filepath).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .unwrap();

    let mut format = probed.format;
    let track = format.default_track().unwrap();
    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .unwrap();

    let mut samples = Vec::new();
    let source_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map_or(1, |c| c.count());

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
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
        let mut sample_buf =
            SampleBuffer::<f32>::new(u64::try_from(num_frames).unwrap_or(u64::MAX), spec);
        sample_buf.copy_interleaved_ref(decoded);

        for frame in sample_buf.samples().chunks(channels) {
            let mono: f32 = frame.iter().sum::<f32>() / channels as f32;
            samples.push(mono);
        }

        let target = (TARGET_SAMPLES as f64 * f64::from(source_rate) / f64::from(SAMPLE_RATE)) as usize;
        if samples.len() >= target {
            break;
        }
    }

    // Simple resample to 48kHz if needed
    if source_rate != SAMPLE_RATE {
        let ratio = f64::from(source_rate) / f64::from(SAMPLE_RATE);
        let output_len = (samples.len() as f64 / ratio) as usize;
        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx as usize;
            let frac = (src_idx - idx0 as f64) as f32;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            output.push(samples[idx0] * (1.0 - frac) + samples[idx1] * frac);
        }
        samples = output;
    }

    samples.resize(TARGET_SAMPLES, 0.0);
    samples
}

/// Sign-quantize to bits (same as protocol::audio_fingerprint but on raw PCM)
fn sign_bits(vec: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; vec.len().div_ceil(8)];
    for (i, &val) in vec.iter().enumerate() {
        if val > 0.0 {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    bytes
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let sum: f64 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    (sum / n as f64).sqrt() as f32
}

fn main() {
    let test_dir = "/tmp/mzk-codec-test";
    let files = [
        ("WAV (original)", "original.wav"),
        ("FLAC (lossless)", "lossless.flac"),
        ("ALAC (lossless)", "alac.m4a"),
        ("AAC 256k", "aac_256.m4a"),
        ("MP3 V0 (~245k)", "mp3_v0.mp3"),
        ("MP3 128k", "mp3_128.mp3"),
    ];

    eprintln!("Decoding {} versions to PCM (48kHz mono, 10s)...\n", files.len());

    let pcm_data: Vec<(&str, Vec<f32>)> = files
        .iter()
        .map(|(label, name)| {
            let path = format!("{test_dir}/{name}");
            eprint!("  {label}...");
            let pcm = decode_to_pcm(&path);
            eprintln!(" {} samples", pcm.len());
            (*label, pcm)
        })
        .collect();

    let reference = &pcm_data[0].1; // WAV is reference

    // --- PCM-level comparison ---
    eprintln!("\n=== PCM-Level Comparison (vs WAV reference) ===\n");
    eprintln!("{:<20} {:>12} {:>12} {:>12}", "Format", "RMS Diff", "Cosine Sim", "Max Diff");
    eprintln!("{}", "-".repeat(60));

    for (label, pcm) in &pcm_data {
        let rms = rms_diff(reference, pcm);
        let cos = cosine_sim(reference, pcm);
        let max_d: f32 = reference
            .iter()
            .zip(pcm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{label:<20} {rms:>12.8} {cos:>12.8} {max_d:>12.8}");
    }

    // --- Sign-bit fingerprint comparison (simulates what CLAP output would look like) ---
    eprintln!("\n=== PCM Sign-Bit Fingerprint (vs WAV, {} bits) ===\n", TARGET_SAMPLES);

    let ref_bits = sign_bits(reference);
    eprintln!("{:<20} {:>12} {:>12}", "Format", "Hamming Dist", "% Bits Diff");
    eprintln!("{}", "-".repeat(48));

    for (label, pcm) in &pcm_data {
        let bits = sign_bits(pcm);
        let dist = hamming(&ref_bits, &bits);
        let pct = dist as f64 / TARGET_SAMPLES as f64 * 100.0;
        eprintln!("{label:<20} {dist:>12} {pct:>11.4}%");
    }

    // --- Coarse fingerprint (blocks of 16) ---
    eprintln!("\n=== Coarse Fingerprint (16-sample blocks, vs WAV) ===\n");

    let block_size = 16;
    let coarsen = |pcm: &[f32]| -> Vec<f32> {
        pcm.chunks(block_size)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };

    let ref_coarse = coarsen(reference);
    let ref_coarse_bits = sign_bits(&ref_coarse);
    let total_coarse_bits = ref_coarse.len();

    eprintln!("{:<20} {:>12} {:>12}", "Format", "Hamming Dist", "% Bits Diff");
    eprintln!("{}", "-".repeat(48));

    for (label, pcm) in &pcm_data {
        let coarse = coarsen(pcm);
        let bits = sign_bits(&coarse);
        let dist = hamming(&ref_coarse_bits, &bits);
        let pct = dist as f64 / total_coarse_bits as f64 * 100.0;
        eprintln!("{label:<20} {dist:>12} {pct:>11.4}%");
    }

    eprintln!("\nNote: These are RAW PCM comparisons. CLAP embeds into 512 dims,");
    eprintln!("so the actual fingerprint divergence will be different (likely smaller");
    eprintln!("because CLAP captures semantic features, not sample-level detail).");
}
