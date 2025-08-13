// --- DATA STRUCTURES ---

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Represents a single point for JSON serialization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializablePoint {
    pub x: i32,
    pub y: i32,
}

/// Represents a complete zone with its metadata for the final JSON output.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Zone {
    pub id: u32,
    pub cam: u32,
    pub mode: String,
    pub field: u32,
    pub points: Vec<SerializablePoint>,
    pub is_closed: bool,
}

/// Holds the metadata for a zone during editing.
#[derive(Debug, Clone)]
pub struct ZoneMetadata {
    pub cam: u32,
    pub mode: String,
    pub field: u32,
}

impl Default for ZoneMetadata {
    fn default() -> Self {
        ZoneMetadata {
            cam: 1,
            mode: "Defense".to_string(),
            field: 0,
        }
    }
}

/// Holds the entire application's state.
#[derive(Debug, Clone)]
pub struct AppState {
    pub zones: HashMap<u32, Vec<opencv::core::Point>>,
    pub closed_status: HashMap<u32, bool>,
    pub metadata: HashMap<u32, ZoneMetadata>,
    pub active_zone_id: u32,
}