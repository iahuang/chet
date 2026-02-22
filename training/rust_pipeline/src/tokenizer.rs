/// FEN tokenization and UCI target encoding, matching chet/tokenizer.py
/// and chet/tokenizer_v3c.py.
///
/// Base vocabulary (16 tokens, v3b):
///   0      = empty square
///   1–6    = white P, N, B, R, Q, K
///   7–12   = black p, n, b, r, q, k
///   13     = white to move
///   14     = black to move
///   15     = [CLS]
///
/// v3c vocabulary (18 tokens — CLS replaced by repetition tokens):
///   0–14   = same as above
///   15     = position seen 1× (replaces CLS)
///   16     = position seen 2×
///   17     = position seen 3+×
///
/// v3b layout (66 tokens): [sq0..sq63] [turn] [CLS]
/// v3c layout (66 tokens): [sq0..sq63] [turn] [rep_count]
///
/// Square indices: a1=0, b1=1, ..., h8=63 (python-chess convention).

const TOKEN_TURN_WHITE: u8 = 13;
const TOKEN_TURN_BLACK: u8 = 14;
const TOKEN_CLS: u8 = 15;
const TOKEN_REP_1: u8 = 15;
const TOKEN_REP_2: u8 = 16;
const TOKEN_REP_3_PLUS: u8 = 17;

/// Lookup table: ASCII byte → piece token ID (0 for non-piece chars).
const fn build_piece_table() -> [u8; 128] {
    let mut t = [0u8; 128];
    t[b'P' as usize] = 1;
    t[b'N' as usize] = 2;
    t[b'B' as usize] = 3;
    t[b'R' as usize] = 4;
    t[b'Q' as usize] = 5;
    t[b'K' as usize] = 6;
    t[b'p' as usize] = 7;
    t[b'n' as usize] = 8;
    t[b'b' as usize] = 9;
    t[b'r' as usize] = 10;
    t[b'q' as usize] = 11;
    t[b'k' as usize] = 12;
    t
}

static PIECE_TABLE: [u8; 128] = build_piece_table();

/// Parse a FEN string into a 66-element token array.
pub fn tokenize_fen(fen: &str) -> [u8; 66] {
    let mut tokens = [0u8; 66];
    let bytes = fen.as_bytes();

    // Find the space separating piece placement from the rest.
    let space = bytes.iter().position(|&b| b == b' ').unwrap_or(bytes.len());

    // Parse piece placement (FEN ranks go 8→1, top to bottom).
    let mut sq: usize = 56; // a8
    for &ch in &bytes[..space] {
        match ch {
            b'/' => sq -= 16, // next rank down: subtract 8 (rank) + 8 (file overshoot)
            b'1'..=b'8' => sq += (ch - b'0') as usize,
            _ => {
                tokens[sq] = PIECE_TABLE[ch as usize];
                sq += 1;
            }
        }
    }

    // Turn token.
    let turn = if space + 1 < bytes.len() {
        bytes[space + 1]
    } else {
        b'w'
    };
    tokens[64] = if turn == b'w' { TOKEN_TURN_WHITE } else { TOKEN_TURN_BLACK };
    tokens[65] = TOKEN_CLS;

    tokens
}

/// Parse a FEN string into a 66-element token array (v3c format).
///
/// Identical to `tokenize_fen` for positions 0–64, but replaces the CLS
/// token at position 65 with a repetition-count token.  Record size stays
/// at 66 bytes.
pub fn tokenize_fen_v3c(fen: &str, repetition_count: u8) -> [u8; 66] {
    let mut tokens = tokenize_fen(fen);

    tokens[65] = match repetition_count {
        0 | 1 => TOKEN_REP_1,
        2 => TOKEN_REP_2,
        _ => TOKEN_REP_3_PLUS,
    };

    tokens
}

/// Encode a UCI move string (e.g. "e2e4") into a target index.
///
/// Returns `from_sq * 64 + to_sq` (range 0–4095).
pub fn encode_uci_target(uci: &str) -> u16 {
    let b = uci.as_bytes();
    let from_sq = (b[1] - b'1') as u16 * 8 + (b[0] - b'a') as u16;
    let to_sq = (b[3] - b'1') as u16 * 8 + (b[2] - b'a') as u16;
    from_sq * 64 + to_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let tokens = tokenize_fen(fen);

        // Rank 1 (a1–h1): R N B Q K B N R
        assert_eq!(tokens[0], 4); // a1 = white R
        assert_eq!(tokens[1], 2); // b1 = white N
        assert_eq!(tokens[2], 3); // c1 = white B
        assert_eq!(tokens[3], 5); // d1 = white Q
        assert_eq!(tokens[4], 6); // e1 = white K
        assert_eq!(tokens[5], 3); // f1 = white B
        assert_eq!(tokens[6], 2); // g1 = white N
        assert_eq!(tokens[7], 4); // h1 = white R

        // Rank 2 (a2–h2): all white pawns
        for sq in 8..16 {
            assert_eq!(tokens[sq], 1, "sq {sq} should be white pawn");
        }

        // Ranks 3–6 (a3–h6): empty
        for sq in 16..48 {
            assert_eq!(tokens[sq], 0, "sq {sq} should be empty");
        }

        // Rank 7 (a7–h7): all black pawns
        for sq in 48..56 {
            assert_eq!(tokens[sq], 7, "sq {sq} should be black pawn");
        }

        // Rank 8 (a8–h8): r n b q k b n r
        assert_eq!(tokens[56], 10); // a8 = black r
        assert_eq!(tokens[57], 8);  // b8 = black n
        assert_eq!(tokens[58], 9);  // c8 = black b
        assert_eq!(tokens[59], 11); // d8 = black q
        assert_eq!(tokens[60], 12); // e8 = black k
        assert_eq!(tokens[61], 9);  // f8 = black b
        assert_eq!(tokens[62], 8);  // g8 = black n
        assert_eq!(tokens[63], 10); // h8 = black r

        // Turn + CLS
        assert_eq!(tokens[64], TOKEN_TURN_WHITE);
        assert_eq!(tokens[65], TOKEN_CLS);
    }

    #[test]
    fn test_black_to_move() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
        let tokens = tokenize_fen(fen);
        assert_eq!(tokens[64], TOKEN_TURN_BLACK);
    }

    #[test]
    fn test_encode_e2e4() {
        // e2 = rank 1 * 8 + file 4 = 12, e4 = rank 3 * 8 + file 4 = 28
        assert_eq!(encode_uci_target("e2e4"), 12 * 64 + 28);
    }

    #[test]
    fn test_encode_a1h8() {
        // a1 = 0, h8 = 63
        assert_eq!(encode_uci_target("a1h8"), 0 * 64 + 63);
    }

    #[test]
    fn test_v3c_rep_tokens() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        let t1 = tokenize_fen_v3c(fen, 1);
        assert_eq!(t1[65], 15); // rep×1 (same value as old CLS)
        assert_eq!(t1.len(), 66);

        let t2 = tokenize_fen_v3c(fen, 2);
        assert_eq!(t2[65], 16); // rep×2

        let t3 = tokenize_fen_v3c(fen, 3);
        assert_eq!(t3[65], 17); // rep×3+

        let t5 = tokenize_fen_v3c(fen, 5);
        assert_eq!(t5[65], 17); // rep×3+ (clamped)

        // First 65 bytes (squares + turn) should be identical to the base tokenizer
        let base = tokenize_fen(fen);
        assert_eq!(&t1[..65], &base[..65]);
    }
}
