use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use pgn_reader::{BufferedReader, RawTag, SanPlus, Skip, Visitor};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::EnPassantMode;
use shakmaty::{Chess, Position};

use crate::sampling::should_keep;
use crate::tokenizer;

const LOG_EVERY_N_GAMES: u64 = 10_000;

/// Visitor that tokenizes positions and writes 68-byte binary records:
///   [66 bytes: tokens] [2 bytes: target (little-endian u16)]
struct PgnVisitor {
    writer: BufWriter<File>,
    pos: Chess,
    move_num: u32,
    games: u64,
    skipped_games: u64,
    kept: u64,
    total: u64,
    label: String,
    /// If set, both players must have Elo >= this value for a game to be included.
    min_elo: Option<u16>,
    /// Tracks Elo values parsed from the current game's headers.
    white_elo: Option<u16>,
    black_elo: Option<u16>,
    /// Whether the current game passes the Elo filter (computed at end of headers).
    skip_game: bool,
}

impl PgnVisitor {
    fn new(out_path: &Path, label: String, min_elo: Option<u16>) -> std::io::Result<Self> {
        let file = File::create(out_path)?;
        let writer = BufWriter::new(file);
        Ok(PgnVisitor {
            writer,
            pos: Chess::default(),
            move_num: 0,
            games: 0,
            skipped_games: 0,
            kept: 0,
            total: 0,
            label,
            min_elo,
            white_elo: None,
            black_elo: None,
            skip_game: false,
        })
    }
}

impl Visitor for PgnVisitor {
    type Result = ();

    fn begin_game(&mut self) {
        self.pos = Chess::default();
        self.move_num = 0;
        self.white_elo = None;
        self.black_elo = None;
        self.skip_game = false;
        self.games += 1;

        if self.games % LOG_EVERY_N_GAMES == 0 {
            eprintln!(
                "  [{}] {:>9} games ({:>9} skipped), {:>9} / {:>9} positions kept",
                self.label,
                format_num(self.games),
                format_num(self.skipped_games),
                format_num(self.kept),
                format_num(self.total),
            );
        }
    }

    fn tag(&mut self, name: &[u8], value: RawTag<'_>) {
        if self.min_elo.is_some() {
            let val_str = String::from_utf8_lossy(value.as_bytes());
            match name {
                b"WhiteElo" => {
                    self.white_elo = val_str.trim().parse::<u16>().ok();
                }
                b"BlackElo" => {
                    self.black_elo = val_str.trim().parse::<u16>().ok();
                }
                _ => {}
            }
        }
    }

    fn end_tags(&mut self) -> Skip {
        if let Some(min) = self.min_elo {
            let dominated = match (self.white_elo, self.black_elo) {
                (Some(w), Some(b)) => w >= min && b >= min,
                // If either Elo is missing, skip the game
                _ => false,
            };
            if !dominated {
                self.skip_game = true;
                self.skipped_games += 1;
                return Skip(true);
            }
        }
        Skip(false)
    }

    fn san(&mut self, san_plus: SanPlus) {
        let m = match san_plus.san.to_move(&self.pos) {
            Ok(m) => m,
            Err(_) => {
                self.move_num += 1;
                return;
            }
        };

        self.total += 1;

        if should_keep(self.move_num) {
            let fen = Fen::from_position(&self.pos, EnPassantMode::Legal).to_string();
            let uci = UciMove::from_standard(m.clone()).to_string();

            let tokens = tokenizer::tokenize_fen(&fen);
            let target = tokenizer::encode_uci_target(&uci);

            let _ = self.writer.write_all(&tokens);
            let _ = self.writer.write_all(&target.to_le_bytes());

            self.kept += 1;
        }

        self.pos.play_unchecked(m);
        self.move_num += 1;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {}
}

/// Process a single PGN file, writing tokenized positions to a temp `.bin` file.
///
/// If `min_elo` is `Some(n)`, only games where both players have Elo >= n are included.
///
/// Returns the path to the temp binary file.
pub fn process_pgn_file(pgn_path: &Path, min_elo: Option<u16>) -> std::io::Result<std::path::PathBuf> {
    let label = pgn_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let out_path = pgn_path.with_extension("bin");
    let mut visitor = PgnVisitor::new(&out_path, label.clone(), min_elo)?;

    let file = File::open(pgn_path)?;
    let reader = BufReader::new(file);
    let mut pgn_reader = BufferedReader::new(reader);

    pgn_reader.read_all(&mut visitor)?;
    visitor.writer.flush()?;

    eprintln!(
        "  [{}] DONE — {} games ({} skipped), {} / {} positions kept",
        label,
        format_num(visitor.games),
        format_num(visitor.skipped_games),
        format_num(visitor.kept),
        format_num(visitor.total),
    );

    Ok(out_path)
}

fn format_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
