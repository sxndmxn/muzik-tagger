use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray, cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};

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

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_fixed_list(vecs: &[Vec<f32>], dim: usize) -> Result<ArrayRef> {
    let mut flat = Vec::with_capacity(vecs.len() * dim);
    for vec in vecs {
        if vec.len() == dim {
            flat.extend_from_slice(vec);
        } else {
            let copy_len = vec.len().min(dim);
            flat.extend_from_slice(&vec[..copy_len]);
            flat.resize(flat.len() + dim - copy_len, 0.0);
        }
    }
    let values = Float32Array::from(flat);
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    Ok(Arc::new(
        FixedSizeListArray::try_new(field, dim as i32, Arc::new(values), None)
            .context("failed to build FixedSizeList array")?,
    ))
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
        build_fixed_list(&meta_vecs, METADATA_DIM)?,
        build_fixed_list(&audio_vecs, AUDIO_DIM)?,
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
        build_fixed_list(&meta_vecs, METADATA_DIM)?,
        build_fixed_list(&audio_vecs, AUDIO_DIM)?,
    ])?;

    Ok(batch)
}

fn runtime() -> Result<tokio::runtime::Handle> {
    static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    if let Ok(h) = tokio::runtime::Handle::try_current() {
        return Ok(h);
    }
    // OnceLock::get_or_try_init is nightly-only, so initialise once and
    // surface the error on the very first call.
    if let Some(rt) = RT.get() {
        return Ok(rt.handle().clone());
    }
    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    // Another thread may have raced us; that's fine — just use whoever won.
    let _ = RT.set(rt);
    Ok(RT.get().context("runtime init race")?.handle().clone())
}

/// Open (or create) the `LanceDB` database.
pub fn open_db() -> Result<Db> {
    let path = db_path();
    std::fs::create_dir_all(&path)?;
    let rt = runtime()?;
    let conn = rt.block_on(lancedb::connect(path.to_string_lossy().as_ref()).execute())?;
    Ok(Db { conn })
}

/// Atomically write a table using `create_table` with overwrite mode.
///
/// `LanceDB` sequential open/delete/add cycles lose data due to version
/// conflicts, so we always write the full table in one shot.
async fn overwrite_table(
    conn: &lancedb::Connection,
    name: &str,
    batch: RecordBatch,
) -> Result<()> {
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    conn.create_table(name, Box::new(batches))
        .mode(lancedb::database::CreateTableMode::Overwrite)
        .execute()
        .await?;
    Ok(())
}

/// Get or create an empty table (for reads when no data has been written yet).
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

/// Get or create an empty albums table.
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

/// Write all tracks atomically (replaces the entire tracks table).
pub fn write_all_tracks(db: &mut Db, tracks: &[Track]) -> Result<()> {
    if tracks.is_empty() {
        return Ok(());
    }
    let rt = runtime()?;
    rt.block_on(async {
        let batch = tracks_to_batch(tracks)?;
        overwrite_table(&db.conn, "tracks", batch).await
    })
}

/// Write all albums atomically (replaces the entire albums table).
pub fn write_all_albums(db: &mut Db, albums: &[Album]) -> Result<()> {
    if albums.is_empty() {
        return Ok(());
    }
    let rt = runtime()?;
    rt.block_on(async {
        let batch = albums_to_batch(albums)?;
        overwrite_table(&db.conn, "albums", batch).await
    })
}

/// Load all tracks from the database.
pub fn all_tracks(db: &Db) -> Result<Vec<Track>> {
    let rt = runtime()?;
    rt.block_on(async {
        let table = get_or_create_tracks(&db.conn).await?;
        let batches: Vec<RecordBatch> = table
            .query()
            .limit(i32::MAX as usize)
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut tracks = Vec::new();
        for batch in &batches {
            tracks.extend(batch_to_tracks(batch)?);
        }
        Ok(tracks)
    })
}

/// Load all albums from the database.
pub fn all_albums(db: &Db) -> Result<Vec<Album>> {
    let rt = runtime()?;
    rt.block_on(async {
        let table = get_or_create_albums(&db.conn).await?;
        let batches: Vec<RecordBatch> = table
            .query()
            .limit(i32::MAX as usize)
            .execute()
            .await?
            .try_collect()
            .await?;
        let mut albums = Vec::new();
        for batch in &batches {
            albums.extend(batch_to_albums(batch)?);
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

fn extract_fixed_list_f32(
    array: &dyn arrow_array::Array,
    row: usize,
) -> Result<Vec<f32>> {
    let list = array.as_fixed_size_list();
    let values = list.value(row);
    let float_arr = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .context("expected Float32Array in fixed size list")?;
    Ok(float_arr.values().to_vec())
}

fn col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .with_context(|| format!("missing {name} column"))
}

fn batch_to_tracks(batch: &RecordBatch) -> Result<Vec<Track>> {
    let num_rows = batch.num_rows();
    let mut tracks = Vec::with_capacity(num_rows);

    let id_col = col(batch, "id")?;
    let title_col = col(batch, "title")?;
    let artist_col = col(batch, "artist")?;
    let album_artist_col = col(batch, "album_artist")?;
    let album_col = col(batch, "album")?;
    let genre_col = col(batch, "genre")?;
    let year_col = col(batch, "year")?;
    let track_num_col = col(batch, "track_num")?;
    let filepath_col = col(batch, "filepath")?;
    let meta_col = col(batch, "metadata_vec")?;
    let audio_col = col(batch, "audio_vec")?;

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
            metadata_vec: extract_fixed_list_f32(meta_col.as_ref(), i)?,
            audio_vec: extract_fixed_list_f32(audio_col.as_ref(), i)?,
        });
    }

    Ok(tracks)
}

fn batch_to_albums(batch: &RecordBatch) -> Result<Vec<Album>> {
    let num_rows = batch.num_rows();
    let mut albums = Vec::with_capacity(num_rows);

    let id_col = col(batch, "id")?;
    let artist_col = col(batch, "artist")?;
    let album_col = col(batch, "album")?;
    let year_col = col(batch, "year")?;
    let genres_col = col(batch, "genres")?;
    let count_col = col(batch, "track_count")?;
    let meta_col = col(batch, "metadata_vec")?;
    let audio_col = col(batch, "audio_vec")?;

    for i in 0..num_rows {
        albums.push(Album {
            id: id_col.as_string::<i32>().value(i).to_string(),
            artist: artist_col.as_string::<i32>().value(i).to_string(),
            album: album_col.as_string::<i32>().value(i).to_string(),
            year: year_col.as_primitive::<arrow_array::types::Int32Type>().value(i),
            genres: genres_col.as_string::<i32>().value(i).to_string(),
            track_count: count_col.as_primitive::<arrow_array::types::Int32Type>().value(i),
            metadata_vec: extract_fixed_list_f32(meta_col.as_ref(), i)?,
            audio_vec: extract_fixed_list_f32(audio_col.as_ref(), i)?,
        });
    }

    Ok(albums)
}
