use rand::Rng;

/// Return the probability of keeping a position at this move number.
/// Move 0: 1%, Move 10: 51%, Move 20+: 100%
pub fn keep_probability(move_num: u32) -> f64 {
    (0.01 + 0.05 * move_num as f64).min(1.0)
}

/// Decide whether to keep a position at this move number.
pub fn should_keep(move_num: u32) -> bool {
    move_num >= 20 || rand::thread_rng().gen::<f64>() < keep_probability(move_num)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keep_probability_boundaries() {
        assert!((keep_probability(0) - 0.01).abs() < f64::EPSILON);
        assert!((keep_probability(10) - 0.51).abs() < f64::EPSILON);
        assert!((keep_probability(20) - 1.0).abs() < f64::EPSILON);
        assert!((keep_probability(30) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_should_keep_always_after_20() {
        for move_num in 20..100 {
            assert!(should_keep(move_num));
        }
    }
}
