# mzk Architecture

## System Overview

```
                            ┌─────────────────────────────────────┐
                            │              mzk CLI                │
                            │  scan │ embed │ similar │ search    │
                            │  radio│ profile│ serve  │ network   │
                            └────────────┬────────────────────────┘
                                         │
                    ┌────────────────────┬┴┬────────────────────┐
                    │                    │  │                    │
              ┌─────▼─────┐      ┌──────▼──▼───┐        ┌──────▼──────┐
              │   scan.rs  │      │ recommend.rs │        │   net/      │
              │            │      │              │        │             │
              │ lofty tags │      │ cosine_sim   │        │ node.rs     │
              │ walkdir    │      │ hybrid_score │        │ protocol.rs │
              │ sha2 ids   │      │ radio        │        │ query.rs    │
              └─────┬──────┘      │ profile      │        └──────┬──────┘
                    │             └──────┬───────┘               │
                    │                    │                        │
              ┌─────▼────────────────────▼────────┐              │
              │             db.rs                  │              │
              │                                    │              │
              │  LanceDB (~/.local/share/mzk/)     │◄─────────────┘
              │  tracks table │ albums table        │
              │  FixedSizeList(384) + (512) vectors │
              │  Arrow RecordBatch serialization     │
              └────────────────┬───────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │           models.rs                 │
              │                                     │
              │  Track { id, title, artist, album,  │
              │    genre, year, metadata_vec[384],   │
              │    audio_vec[512], filepath }        │
              │  Album { id, artist, album, year,   │
              │    genres, track_count, vecs }       │
              │  SearchResult { score, artist,       │
              │    title, album, year, filepath }    │
              └─────────────────────────────────────┘
```

## Embedding Pipeline

```
  Audio File (.mp3/.flac/...)
       │
       ├──► lofty ──► metadata tags ──► Track struct
       │
       ├──► embed/metadata.rs
       │      │
       │      ├─ tokenizers (HuggingFace) ──► token IDs
       │      └─ ort (all-MiniLM-L6-v2.onnx) ──► 384-dim vector
       │           └─ mean pooling + L2 normalize
       │
       └──► embed/audio.rs
              │
              ├─ symphonia ──► PCM f32 mono @ 48kHz (10s)
              └─ ort (clap-htsat-tiny.onnx) ──► 512-dim vector
                   └─ L2 normalize
```

## Similarity Search Flow

```
  mzk similar "Artist" "Title" -n 50

  1. db::find_track()       ──► full table scan, match by name
  2. db::all_tracks()       ──► load entire DB into memory
  3. for each candidate:
       hybrid_score = (1-w) * cosine(meta_q, meta_c)
                    +   w   * cosine(audio_q, audio_c)
       where w = 0.7 (default audio weight)
  4. sort by score descending
  5. truncate to n
  6. stdout: TSV or NDJSON
```

## P2P Network Architecture

```
  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
  │   Node A     │         │   Node B     │         │   Node C     │
  │              │         │              │         │              │
  │ ┌──────────┐ │  TCP/   │ ┌──────────┐ │  TCP/   │ ┌──────────┐ │
  │ │ LanceDB  │ │  QUIC   │ │ LanceDB  │ │  QUIC   │ │ LanceDB  │ │
  │ │ (local)  │ │◄───────►│ │ (local)  │ │◄───────►│ │ (local)  │ │
  │ └──────────┘ │  Noise  │ └──────────┘ │  Noise  │ └──────────┘ │
  │              │  crypto  │              │  crypto  │              │
  │ libp2p:      │         │ libp2p:      │         │ libp2p:      │
  │  - Kademlia  │         │  - Kademlia  │         │  - Kademlia  │
  │  - Identify  │         │  - Identify  │         │  - Identify  │
  │  - Req/Resp  │         │  - Req/Resp  │         │  - Req/Resp  │
  └──────────────┘         └──────────────┘         └──────────────┘

  Discovery: Kademlia DHT (bootstrap → find peers → exchange addrs)
  Transport: TCP + QUIC, encrypted via Noise protocol
  Muxing:    Yamux (multiple streams per connection)

  Query Protocol: /mzk/search/1.0.0 (msgpack serialization)
  ┌─────────────────────────────┐    ┌─────────────────────────────┐
  │ SearchQuery                 │    │ SearchResponse              │
  │  ├─ TextSearch              │    │  └─ results: [PeerResult]   │
  │  │   ├─ query_vec: [f32]    │──►│       ├─ score: f32         │
  │  │   └─ max_results: usize  │    │       ├─ artist, title     │
  │  └─ SimilarSearch           │    │       ├─ album, year       │
  │      ├─ metadata_vec        │    │       └─ peer_id           │
  │      ├─ audio_vec           │    └─────────────────────────────┘
  │      ├─ weight_audio: f32   │
  │      └─ max_results: usize  │
  └─────────────────────────────┘

  Federated Query Flow (PLANNED - not yet wired):
  1. User issues query on Node A
  2. Node A finds K nearest peers via Kademlia
  3. Sends SearchQuery to each peer
  4. Each peer runs local similarity search
  5. Node A collects SearchResponses (with timeout)
  6. Merge + deduplicate by (artist, title, album)
  7. Re-rank by score, return top N
```

## Data Flow Summary

```
  ONLY traverses network:          NEVER leaves node:
  ┌─────────────────────┐         ┌─────────────────────┐
  │ - embedding vectors │         │ - audio files        │
  │ - track metadata    │         │ - file paths         │
  │   (artist, title,   │         │ - full database      │
  │    album, year)     │         │ - ONNX models        │
  │ - similarity scores │         │                      │
  └─────────────────────┘         └─────────────────────┘
```

## Module Dependency Graph

```
  cli.rs ─────────────────────────────────────────► models.rs
    │                                                   ▲
    ├──► scan.rs ──────► db.rs ─────────────────────────┤
    │                      ▲                            │
    ├──► embed/mod.rs ─────┤                            │
    │      ├─ metadata.rs ─┤                            │
    │      └─ audio.rs ────┘                            │
    │                                                   │
    ├──► recommend.rs ──► db.rs                         │
    │                                                   │
    └──► net/mod.rs                                     │
           ├─ protocol.rs ──────────────────────────────┘
           ├─ node.rs ──────► recommend.rs, db.rs
           └─ query.rs ──────► protocol.rs
```
