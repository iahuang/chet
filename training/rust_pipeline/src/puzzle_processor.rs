use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::EnPassantMode;
use shakmaty::{CastlingMode, Chess, Position};

use crate::tokenizer;

/// Process a zstd-compressed puzzle CSV, writing tokenized positions to a temp `.bin` file.
///
/// Each puzzle's move sequence alternates: opponent move, player move, opponent, player, ...
/// Every player move (odd indices: 1, 3, 5, ...) produces a 67-byte binary record
/// with the tokenized FEN and the move target. No sampling is applied to puzzles.
///
/// Returns the path to the temp binary file and the number of positions written.
pub fn process_puzzles(
    puzzle_path: &Path,
    max_puzzles: Option<usize>,
) -> std::io::Result<(std::path::PathBuf, usize)> {
    let out_path = puzzle_path.with_extension("bin");
    let file = File::open(puzzle_path)?;
    let decoder = zstd::Decoder::new(BufReader::new(file))?;
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(decoder);

    let out_file = File::create(&out_path)?;
    let mut writer = BufWriter::new(out_file);

    let mut count: usize = 0;
    let mut puzzle_count: usize = 0;
    let mut errors: u64 = 0;

    for result in csv_reader.records() {
        if let Some(max) = max_puzzles {
            if puzzle_count >= max {
                break;
            }
        }

        let record = match result {
            Ok(r) => r,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        // Columns: 0=PuzzleId, 1=FEN, 2=Moves, 3=Rating, 4=RatingDeviation,
        //          5=Popularity, 6=NbPlays, 7=Themes, 8=GameUrl, 9=OpeningTags
        let fen_str = match record.get(1) {
            Some(v) => v,
            None => continue,
        };
        let moves_str = match record.get(2) {
            Some(v) => v,
            None => continue,
        };

        let moves: Vec<&str> = moves_str.split_whitespace().collect();
        if moves.len() < 2 {
            continue;
        }

        // Parse starting FEN
        let fen: Fen = match fen_str.parse() {
            Ok(f) => f,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        let mut pos: Chess = match fen.into_position(CastlingMode::Standard) {
            Ok(p) => p,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        // Walk through all moves. Odd-indexed moves (1, 3, 5, ...) are player moves.
        let mut valid = true;
        for (i, move_str) in moves.iter().enumerate() {
            let uci: UciMove = match move_str.parse() {
                Ok(m) => m,
                Err(_) => {
                    errors += 1;
                    valid = false;
                    break;
                }
            };

            let m = match uci.to_move(&pos) {
                Ok(m) => m,
                Err(_) => {
                    errors += 1;
                    valid = false;
                    break;
                }
            };

            if i % 2 == 1 {
                // Player move — tokenize and write binary record
                let puzzle_fen =
                    Fen::from_position(&pos, EnPassantMode::Legal).to_string();
                let tokens = tokenizer::tokenize_fen(&puzzle_fen);
                let target = tokenizer::encode_uci_target(move_str);

                writer.write_all(&tokens)?;
                writer.write_all(&target.to_le_bytes())?;
                count += 1;
            }

            pos.play_unchecked(m);
        }

        if valid {
            puzzle_count += 1;
        }

        if puzzle_count % 10_000 == 0 && puzzle_count > 0 {
            eprintln!("  Puzzles: {} puzzles, {} positions", puzzle_count, count);
        }
    }

    writer.flush()?;

    if errors > 0 {
        eprintln!("  Puzzles: {} parse errors skipped", errors);
    }

    eprintln!(
        "  Puzzles: DONE — {} puzzles, {} positions written",
        puzzle_count, count
    );

    Ok((out_path, count))
}
