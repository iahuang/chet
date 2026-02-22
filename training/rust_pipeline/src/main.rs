mod config;
mod downloader;
mod npy;
mod pgn_processor;
mod puzzle_processor;
mod sampling;
mod tokenizer;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use walkdir::WalkDir;

/// Size of one binary record: token bytes + 2 target bytes.
/// v3b: 66 + 2 = 68, v3c: 67 + 2 = 69.
const RECORD_SIZE_V3B: usize = 68;
const RECORD_SIZE_V3C: usize = 69;

const TOKEN_WIDTH_V3B: usize = 66;
const TOKEN_WIDTH_V3C: usize = 67;

#[derive(Parser)]
#[command(name = "chet-data-pipeline")]
#[command(about = "Fast chess training data pipeline for Chet")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download data sources defined in a TOML config file
    Download {
        /// Path to data_sources.toml
        #[arg(long, value_name = "FILE")]
        config: PathBuf,

        /// Directory to store downloaded/extracted files
        #[arg(long, value_name = "DIR")]
        data_dir: PathBuf,
    },

    /// Process downloaded PGN and puzzle files into training-ready numpy arrays
    Process {
        /// Directory containing raw PGN files (searched recursively)
        #[arg(long, value_name = "DIR")]
        input_dir: PathBuf,

        /// Output directory for tokens.npy, targets.npy, metadata.json
        #[arg(long, value_name = "DIR")]
        output_dir: PathBuf,

        /// Path to zstd-compressed puzzle CSV (optional)
        #[arg(long, value_name = "FILE")]
        puzzle_file: Option<PathBuf>,

        /// Maximum number of puzzles to include
        #[arg(long, default_value = "5000000")]
        max_puzzles: usize,

        /// Minimum Elo for both players; games where either player is below this are skipped
        #[arg(long, value_name = "ELO")]
        min_elo: Option<u16>,

        /// Skip pre-shuffling the data
        #[arg(long)]
        no_shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Emit v3c tokens (67 bytes: 66 base + 1 repetition-count token)
        #[arg(long)]
        v3c: bool,
    },

    /// Download and then process in one step
    Run {
        /// Path to data_sources.toml
        #[arg(long, value_name = "FILE")]
        config: PathBuf,

        /// Directory to store downloaded/extracted files
        #[arg(long, value_name = "DIR")]
        data_dir: PathBuf,

        /// Output directory for tokens.npy, targets.npy, metadata.json
        #[arg(long, value_name = "DIR")]
        output_dir: PathBuf,

        /// Maximum number of puzzles to include
        #[arg(long, default_value = "100000")]
        max_puzzles: usize,

        /// Minimum Elo for both players; games where either player is below this are skipped
        #[arg(long, value_name = "ELO")]
        min_elo: Option<u16>,

        /// Skip pre-shuffling the data
        #[arg(long)]
        no_shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Emit v3c tokens (67 bytes: 66 base + 1 repetition-count token)
        #[arg(long)]
        v3c: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Download { config, data_dir } => {
            let cfg = config::load_config(&config)?;
            cmd_download(&cfg, &data_dir)?;
        }
        Command::Process {
            input_dir,
            output_dir,
            puzzle_file,
            max_puzzles,
            min_elo,
            no_shuffle,
            seed,
            v3c,
        } => {
            cmd_process(
                &input_dir,
                &output_dir,
                puzzle_file.as_deref(),
                max_puzzles,
                min_elo,
                !no_shuffle,
                seed,
                v3c,
            )?;
        }
        Command::Run {
            config,
            data_dir,
            output_dir,
            max_puzzles,
            min_elo,
            no_shuffle,
            seed,
            v3c,
        } => {
            let cfg = config::load_config(&config)?;
            cmd_download(&cfg, &data_dir)?;
            let puzzle_file = downloader::puzzle_file_path(&cfg, &data_dir);
            cmd_process(
                &data_dir,
                &output_dir,
                puzzle_file.as_deref(),
                max_puzzles,
                min_elo,
                !no_shuffle,
                seed,
                v3c,
            )?;
        }
    }

    Ok(())
}

fn cmd_download(
    cfg: &config::DataSources,
    data_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("============================================================");
    eprintln!("DOWNLOADING DATA");
    eprintln!("============================================================");
    downloader::download_all(cfg, data_dir)?;
    eprintln!("Downloads complete.");
    Ok(())
}

fn cmd_process(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    puzzle_file: Option<&std::path::Path>,
    max_puzzles: usize,
    min_elo: Option<u16>,
    shuffle: bool,
    seed: u64,
    v3c: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let record_size = if v3c { RECORD_SIZE_V3C } else { RECORD_SIZE_V3B };
    let token_width = if v3c { TOKEN_WIDTH_V3C } else { TOKEN_WIDTH_V3B };

    eprintln!("============================================================");
    eprintln!("PROCESSING DATA{}", if v3c { " (v3c — with repetition tokens)" } else { "" });
    eprintln!("============================================================");
    if let Some(elo) = min_elo {
        eprintln!("Minimum Elo filter: {}", elo);
    }

    fs::create_dir_all(output_dir)?;

    // --- Discover and process PGN files in parallel ---
    let mut pgn_files: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "pgn")
                .unwrap_or(false)
        })
        .map(|e| e.into_path())
        .collect();

    pgn_files.sort();
    eprintln!("Found {} PGN files", pgn_files.len());

    let temp_bins: Vec<PathBuf> = pgn_files
        .par_iter()
        .filter_map(|pgn_path| {
            eprintln!("Processing: {}", pgn_path.display());
            match pgn_processor::process_pgn_file(pgn_path, min_elo, v3c) {
                Ok(bin_path) => Some(bin_path),
                Err(e) => {
                    eprintln!("Error processing {}: {}", pgn_path.display(), e);
                    None
                }
            }
        })
        .collect();

    // --- Process puzzles ---
    let puzzle_bin = if let Some(puzzle_path) = puzzle_file {
        if puzzle_path.exists() {
            eprintln!("Processing puzzles from {}", puzzle_path.display());
            match puzzle_processor::process_puzzles(puzzle_path, Some(max_puzzles), v3c) {
                Ok((bin_path, _count)) => Some(bin_path),
                Err(e) => {
                    eprintln!("Error processing puzzles: {}", e);
                    None
                }
            }
        } else {
            eprintln!("Puzzle file not found: {}", puzzle_path.display());
            None
        }
    } else {
        None
    };

    // --- Collect all temp .bin paths ---
    let mut all_bins: Vec<PathBuf> = temp_bins;
    if let Some(ref p) = puzzle_bin {
        all_bins.push(p.clone());
    }

    // --- Read all records into memory ---
    eprintln!("Reading temp files...");
    let mut raw_data: Vec<u8> = Vec::new();
    for bin_path in &all_bins {
        let data = fs::read(bin_path)?;
        raw_data.extend_from_slice(&data);
        fs::remove_file(bin_path)?;
    }

    let n = raw_data.len() / record_size;
    assert_eq!(
        raw_data.len() % record_size,
        0,
        "Binary data size is not a multiple of record size ({})",
        record_size,
    );
    eprintln!("Total positions: {}", n);

    // --- Shuffle if requested ---
    if shuffle {
        eprintln!("Shuffling {} records (seed={})...", n, seed);
        shuffle_records(&mut raw_data, n, record_size, seed);
    }

    // --- Write .npy files ---
    let tokens_path = output_dir.join("tokens.npy");
    let targets_path = output_dir.join("targets.npy");

    eprintln!("Writing {}...", tokens_path.display());
    {
        let mut f = BufWriter::new(File::create(&tokens_path)?);
        npy::write_u8_header(&mut f, n, token_width)?;
        for chunk in raw_data.chunks_exact(record_size) {
            f.write_all(&chunk[..token_width])?;
        }
        f.flush()?;
    }

    eprintln!("Writing {}...", targets_path.display());
    {
        let mut f = BufWriter::new(File::create(&targets_path)?);
        npy::write_u16_header(&mut f, n)?;
        for chunk in raw_data.chunks_exact(record_size) {
            f.write_all(&chunk[token_width..token_width + 2])?;
        }
        f.flush()?;
    }

    // --- Write metadata ---
    let metadata_path = output_dir.join("metadata.json");
    let metadata = format!(
        "{{\n  \"num_positions\": {},\n  \"token_width\": {}\n}}\n",
        n, token_width
    );
    fs::write(&metadata_path, metadata)?;

    let tokens_size = fs::metadata(&tokens_path)?.len();
    let targets_size = fs::metadata(&targets_path)?.len();
    eprintln!(
        "\nDone! Wrote {} positions to {}",
        n,
        output_dir.display()
    );
    eprintln!("  tokens.npy:  {:.2} GB", tokens_size as f64 / 1e9);
    eprintln!("  targets.npy: {:.1} MB", targets_size as f64 / 1e6);

    Ok(())
}

/// Shuffle fixed-size records in-place.
fn shuffle_records(raw_data: &mut Vec<u8>, n: usize, record_size: usize, seed: u64) {
    // Fisher-Yates shuffle on variable-size records via index permutation + copy.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut shuffled = vec![0u8; raw_data.len()];
    for (dst, &src) in indices.iter().enumerate() {
        let src_start = src * record_size;
        let dst_start = dst * record_size;
        shuffled[dst_start..dst_start + record_size]
            .copy_from_slice(&raw_data[src_start..src_start + record_size]);
    }
    *raw_data = shuffled;
}
