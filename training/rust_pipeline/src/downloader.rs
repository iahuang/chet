use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};

use crate::config::DataSources;

/// Download all data sources described in the config to `data_dir`.
pub fn download_all(config: &DataSources, data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(data_dir)?;

    for url in &config.pgn_sources {
        let filename = url_filename(url);

        if filename.ends_with(".zip") {
            let extract_name = filename.trim_end_matches(".zip");
            let extract_dir = data_dir.join(extract_name);

            if extract_dir.exists() {
                eprintln!("  {} already exists, skipping", extract_dir.display());
                continue;
            }

            let zip_path = data_dir.join(&filename);
            download_file(url, &zip_path)?;
            extract_zip(&zip_path, &extract_dir)?;
            fs::remove_file(&zip_path)?;
            eprintln!("  Deleted {}", zip_path.display());
        } else {
            let dest = data_dir.join(&filename);
            download_file(url, &dest)?;
        }
    }

    if let Some(puzzle_url) = &config.puzzle_source {
        let filename = url_filename(puzzle_url);
        let dest = data_dir.join(&filename);
        download_file(puzzle_url, &dest)?;
    }

    Ok(())
}

/// Download a file from `url` to `dest`, skipping if it already exists.
fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if dest.exists() {
        eprintln!("  {} already exists, skipping download", dest.display());
        return Ok(());
    }

    eprintln!("  Downloading {}", url);

    let resp = reqwest::blocking::get(url)?.error_for_status()?;
    let total_size = resp.content_length();

    let pb = if let Some(size) = total_size {
        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("##-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("  {spinner} {bytes} ({bytes_per_sec})")
                .unwrap(),
        );
        pb
    };

    let mut file = File::create(dest)?;
    let mut reader = resp;
    let mut buf = [0u8; 8192];

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }

    pb.finish_and_clear();
    eprintln!("  Saved {}", dest.display());
    Ok(())
}

/// Extract a zip archive into `dest_dir`.
fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("  Extracting {}", zip_path.display());
    fs::create_dir_all(dest_dir)?;

    let file = File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_owned();

        let out_path = dest_dir.join(&name);

        if entry.is_dir() {
            fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = File::create(&out_path)?;
            io::copy(&mut entry, &mut out_file)?;
        }
    }

    Ok(())
}

/// Extract the filename from a URL.
fn url_filename(url: &str) -> String {
    url.rsplit('/').next().unwrap_or("download").to_string()
}

/// Return the path to the puzzle file inside `data_dir`, derived from the config URL.
pub fn puzzle_file_path(config: &DataSources, data_dir: &Path) -> Option<PathBuf> {
    config
        .puzzle_source
        .as_ref()
        .map(|url| data_dir.join(url_filename(url)))
}
