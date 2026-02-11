use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct DataSources {
    pub pgn_sources: Vec<String>,
    pub puzzle_source: Option<String>,
}

pub fn load_config(path: &Path) -> Result<DataSources, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let config: DataSources = toml::from_str(&contents)?;
    Ok(config)
}
