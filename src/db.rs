use std::sync::Arc;

use anyhow::Result;
use arrow_array::{
    ArrayRef, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
    builder::Float32Builder,
    builder::ListBuilder,
    cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;

use crate::models::{AUDIO_DIM, Album, METADATA_DIM, Track};

/// The `LanceDB` connection wrapper.
pub struct Db {
    conn: lancedb::Connection,
}

fn db_path() -> std::path::PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("mzk")
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn tracks_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("artist", DataType::Utf8, false),
        Field::new("album_artist", DataType::Utf8, false),
        Field::new("album", DataType::Utf8, false),
        Field::new("genre", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("track_num", DataType::Int32, false),
        Field::new("filepath", DataType::Utf8, false),
        Field::new(
            "metadata_vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                METADATA_DIM as i32,
            ),
            false,
        ),
        Field::new(
            "audio_vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                AUDIO_DIM as i32,
            ),
            false,
        ),
    ]))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn albums_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("artist", DataType::Utf8, false),
        Field::new("album", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("genres", DataType::Utf8, false),
        Field::new("track_count", DataType::Int32, false),
        Field::new(
            "metadata_vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                METADATA_DIM as i32,
            ),
            false,
        ),
        Field::new(
            "audio_vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                AUDIO_DIM as i32,
            ),
            false,
        ),
    ]))
}

fn build_fixed_list(vecs: &[Vec<f32>], dim: usize) -> ArrayRef {
    let mut builder = ListBuilder::with_capacity(Float32Builder::new(), vecs.len())
        .with_field(Field::new("item", DataType::Float32, true));
    for vec in vecs {
        let values: Vec<f32> = if vec.len() == dim {
            vec.clone()
        } else {
            let mut v = vec![0.0f32; dim];
            let copy_len = vec.len().min(dim);
            v[..copy_len].copy_from_slice(&vec[..copy_len]);
            v
        };
        for &val in &values {
            builder.values().append_value(val);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn tracks_to_batch(tracks: &[Track]) -> Result<RecordBatch> {
    let schema = tracks_schema();

    let ids: Vec<&str> = tracks.iter().map(|t| t.id.as_str()).collect();
    let titles: Vec<&str> = tracks.iter().map(|t| t.title.as_str()).collect();
    let artists: Vec<&str> = tracks.iter().map(|t| t.artist.as_str()).collect();
    let album_artists: Vec<&str> = tracks.iter().map(|t| t.album_artist.as_str()).collect();
    let albums: Vec<&str> = tracks.iter().map(|t| t.album.as_str()).collect();
    let genres: Vec<&str> = tracks.iter().map(|t| t.genre.as_str()).collect();
    let years: Vec<i32> = tracks.iter().map(|t| t.year).collect();
    let track_nums: Vec<i32> = tracks.iter().map(|t| t.track_num).collect();
    let filepaths: Vec<&str> = tracks.iter().map(|t| t.filepath.as_str()).collect();
    let meta_vecs: Vec<Vec<f32>> = tracks.iter().map(|t| t.metadata_vec.clone()).collect();
    let audio_vecs: Vec<Vec<f32>> = tracks.iter().map(|t| t.audio_vec.clone()).collect();

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(titles)),
        Arc::new(StringArray::from(artists)),
        Arc::new(StringArray::from(album_artists)),
        Arc::new(StringArray::from(albums)),
        Arc::new(StringArray::from(genres)),
        Arc::new(Int32Array::from(years)),
        Arc::new(Int32Array::from(track_nums)),
        Arc::new(StringArray::from(filepaths)),
        build_fixed_list(&meta_vecs, METADATA_DIM),
        build_fixed_list(&audio_vecs, AUDIO_DIM),
    ])?;

    Ok(batch)
}

fn albums_to_batch(albums: &[Album]) -> Result<RecordBatch> {
    let schema = albums_schema();

    let ids: Vec<&str> = albums.iter().map(|a| a.id.as_str()).collect();
    let artists: Vec<&str> = albums.iter().map(|a| a.artist.as_str()).collect();
    let album_names: Vec<&str> = albums.iter().map(|a| a.album.as_str()).collect();
    let years: Vec<i32> = albums.iter().map(|a| a.year).collect();
    let genres: Vec<&str> = albums.iter().map(|a| a.genres.as_str()).collect();
    let counts: Vec<i32> = albums.iter().map(|a| a.track_count).collect();
    let meta_vecs: Vec<Vec<f32>> = albums.iter().map(|a| a.metadata_vec.clone()).collect();
    let audio_vecs: Vec<Vec<f32>> = albums.iter().map(|a| a.audio_vec.clone()).collect();

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(artists)),
        Arc::new(StringArray::from(album_names)),
        Arc::new(Int32Array::from(years)),
        Arc::new(StringArray::from(genres)),
        Arc::new(Int32Array::from(counts)),
        build_fixed_list(&meta_vecs, METADATA_DIM),
        build_fixed_list(&audio_vecs, AUDIO_DIM),
    ])?;

    Ok(batch)
}

fn runtime() -> tokio::runtime::Handle {
    tokio::runtime::Handle::try_current().unwrap_or_else(|_| {
        tokio::runtime::Runtime::new()
            .expect("failed to create runtime")
            .handle()
            .clone()
    })
}

/// Open (or create) the `LanceDB` database.
pub fn open_db() -> Result<Db> {
    let path = db_path();
    std::fs::create_dir_all(&path)?;
    let rt = runtime();
    let conn = rt.block_on(lancedb::connect(path.to_string_lossy().as_ref()).execute())?;
    Ok(Db { conn })
}

/// Get or create the tracks table.
async fn get_or_create_tracks(conn: &lancedb::Connection) -> Result<lancedb::Table> {
    let tables = conn.table_names().execute().await?;
    if tables.iter().any(|n| n == "tracks") {
        Ok(conn.open_table("tracks").execute().await?)
    } else {
        let schema = tracks_schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let batches = RecordBatchIterator::new(vec![Ok(empty)], schema);
        Ok(conn
            .create_table("tracks", Box::new(batches))
            .execute()
            .await?)
    }
}

/// Get or create the albums table.
async fn get_or_create_albums(conn: &lancedb::Connection) -> Result<lancedb::Table> {
    let tables = conn.table_names().execute().await?;
    if tables.iter().any(|n| n == "albums") {
        Ok(conn.open_table("albums").execute().await?)
    } else {
        let schema = albums_schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let batches = RecordBatchIterator::new(vec![Ok(empty)], schema);
        Ok(conn
            .create_table("albums", Box::new(batches))
            .execute()
            .await?)
    }
}

/// Upsert tracks into the database.
pub fn upsert_tracks(db: &mut Db, tracks: &[Track]) -> Result<()> {
    if tracks.is_empty() {
        return Ok(());
    }
    let rt = runtime();
    rt.block_on(async {
        let table = get_or_create_tracks(&db.conn).await?;
        let batch = tracks_to_batch(tracks)?;
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(Box::new(batches)).await?;
        Ok(())
    })
}

/// Upsert albums into the database.
pub fn upsert_albums(db: &mut Db, albums: &[Album]) -> Result<()> {
    if albums.is_empty() {
        return Ok(());
    }
    let rt = runtime();
    rt.block_on(async {
        let table = get_or_create_albums(&db.conn).await?;
        let batch = albums_to_batch(albums)?;
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(Box::new(batches)).await?;
        Ok(())
    })
}

/// Load all tracks from the database.
pub fn all_tracks(db: &Db) -> Result<Vec<Track>> {
    let rt = runtime();
    rt.block_on(async {
        let table = get_or_create_tracks(&db.conn).await?;
        let batches: Vec<RecordBatch> = table
            .query()
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut tracks = Vec::new();
        for batch in &batches {
            tracks.extend(batch_to_tracks(batch));
        }
        Ok(tracks)
    })
}

/// Load all albums from the database.
pub fn all_albums(db: &Db) -> Result<Vec<Album>> {
    let rt = runtime();
    rt.block_on(async {
        let table = get_or_create_albums(&db.conn).await?;
        let batches: Vec<RecordBatch> = table
            .query()
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut albums = Vec::new();
        for batch in &batches {
            albums.extend(batch_to_albums(batch));
        }
        Ok(albums)
    })
}

/// Find a track by artist and title (case-insensitive).
pub fn find_track(db: &Db, artist: &str, title: &str) -> Result<Option<Track>> {
    let tracks = all_tracks(db)?;
    let artist_lower = artist.to_lowercase();
    let title_lower = title.to_lowercase();
    Ok(tracks
        .into_iter()
        .find(|t| t.artist.to_lowercase() == artist_lower && t.title.to_lowercase() == title_lower))
}

/// Find an album by artist and album name (case-insensitive).
pub fn find_album(db: &Db, artist: &str, album: &str) -> Result<Option<Album>> {
    let albums = all_albums(db)?;
    let artist_lower = artist.to_lowercase();
    let album_lower = album.to_lowercase();
    Ok(albums
        .into_iter()
        .find(|a| a.artist.to_lowercase() == artist_lower && a.album.to_lowercase() == album_lower))
}

/// Update tracks in-place (for embedding updates).
pub fn update_tracks(db: &mut Db, tracks: &[Track]) -> Result<()> {
    upsert_tracks(db, tracks)
}

/// Update albums in-place (for embedding updates).
pub fn update_albums(db: &mut Db, albums: &[Album]) -> Result<()> {
    upsert_albums(db, albums)
}

fn extract_fixed_list_f32(
    array: &dyn arrow_array::Array,
    row: usize,
) -> Vec<f32> {
    let list = array.as_fixed_size_list();
    let values = list.value(row);
    let float_arr = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("expected Float32Array in fixed size list");
    float_arr.values().to_vec()
}

fn batch_to_tracks(batch: &RecordBatch) -> Vec<Track> {
    let num_rows = batch.num_rows();
    let mut tracks = Vec::with_capacity(num_rows);

    let id_col = batch.column_by_name("id").expect("missing id column");
    let title_col = batch.column_by_name("title").expect("missing title column");
    let artist_col = batch.column_by_name("artist").expect("missing artist column");
    let album_artist_col = batch.column_by_name("album_artist").expect("missing album_artist column");
    let album_col = batch.column_by_name("album").expect("missing album column");
    let genre_col = batch.column_by_name("genre").expect("missing genre column");
    let year_col = batch.column_by_name("year").expect("missing year column");
    let track_num_col = batch.column_by_name("track_num").expect("missing track_num column");
    let filepath_col = batch.column_by_name("filepath").expect("missing filepath column");
    let meta_col = batch.column_by_name("metadata_vec").expect("missing metadata_vec column");
    let audio_col = batch.column_by_name("audio_vec").expect("missing audio_vec column");

    for i in 0..num_rows {
        tracks.push(Track {
            id: id_col.as_string::<i32>().value(i).to_string(),
            title: title_col.as_string::<i32>().value(i).to_string(),
            artist: artist_col.as_string::<i32>().value(i).to_string(),
            album_artist: album_artist_col.as_string::<i32>().value(i).to_string(),
            album: album_col.as_string::<i32>().value(i).to_string(),
            genre: genre_col.as_string::<i32>().value(i).to_string(),
            year: year_col.as_primitive::<arrow_array::types::Int32Type>().value(i),
            track_num: track_num_col.as_primitive::<arrow_array::types::Int32Type>().value(i),
            filepath: filepath_col.as_string::<i32>().value(i).to_string(),
            metadata_vec: extract_fixed_list_f32(meta_col.as_ref(), i),
            audio_vec: extract_fixed_list_f32(audio_col.as_ref(), i),
        });
    }

    tracks
}

fn batch_to_albums(batch: &RecordBatch) -> Vec<Album> {
    let num_rows = batch.num_rows();
    let mut albums = Vec::with_capacity(num_rows);

    let id_col = batch.column_by_name("id").expect("missing id column");
    let artist_col = batch.column_by_name("artist").expect("missing artist column");
    let album_col = batch.column_by_name("album").expect("missing album column");
    let year_col = batch.column_by_name("year").expect("missing year column");
    let genres_col = batch.column_by_name("genres").expect("missing genres column");
    let count_col = batch.column_by_name("track_count").expect("missing track_count column");
    let meta_col = batch.column_by_name("metadata_vec").expect("missing metadata_vec column");
    let audio_col = batch.column_by_name("audio_vec").expect("missing audio_vec column");

    for i in 0..num_rows {
        albums.push(Album {
            id: id_col.as_string::<i32>().value(i).to_string(),
            artist: artist_col.as_string::<i32>().value(i).to_string(),
            album: album_col.as_string::<i32>().value(i).to_string(),
            year: year_col.as_primitive::<arrow_array::types::Int32Type>().value(i),
            genres: genres_col.as_string::<i32>().value(i).to_string(),
            track_count: count_col.as_primitive::<arrow_array::types::Int32Type>().value(i),
            metadata_vec: extract_fixed_list_f32(meta_col.as_ref(), i),
            audio_vec: extract_fixed_list_f32(audio_col.as_ref(), i),
        });
    }

    albums
}
