/// Minimal numpy .npy file writer.
///
/// The .npy format (v1.0):
///   6 bytes  magic: \x93NUMPY
///   2 bytes  version: 1, 0
///   2 bytes  header_len (little-endian)
///   N bytes  header dict (ASCII, padded with spaces, terminated with \n)
///   ...      raw data
///
/// The total of (magic + version + header_len + header) is aligned to 64 bytes.

use std::io::{self, Write};

/// Write a .npy header for a uint8 array with shape (n, `token_width`).
pub fn write_u8_header<W: Write>(w: &mut W, n: usize, token_width: usize) -> io::Result<()> {
    write_header(w, "|u1", &format!("({}, {})", n, token_width))
}

/// Write a .npy header for a uint16 (little-endian) array with shape (n,).
pub fn write_u16_header<W: Write>(w: &mut W, n: usize) -> io::Result<()> {
    write_header(w, "<u2", &format!("({},)", n))
}

fn write_header<W: Write>(w: &mut W, descr: &str, shape: &str) -> io::Result<()> {
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape
    );

    // prefix = magic(6) + version(2) + header_len_field(2) = 10
    let prefix_len = 10;
    let raw_len = header_dict.len() + 1; // +1 for trailing \n
    let total = prefix_len + raw_len;
    let padded_total = (total + 63) & !63; // round up to multiple of 64
    let padding = padded_total - total;
    let header_len = (raw_len + padding) as u16;

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1, 0])?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(header_dict.as_bytes())?;
    for _ in 0..padding {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;

    Ok(())
}
