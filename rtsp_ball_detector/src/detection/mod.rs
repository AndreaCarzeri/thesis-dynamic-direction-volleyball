pub mod bounding_box;
pub mod yolo;
pub mod background_subtractor;
pub mod evaluation;
pub mod compare_models;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Detection {
	pub bbox: bounding_box::BoundingBox,
	pub confidence: Option<f32>,
	pub class_id: i32,
}

impl Detection {
    pub fn new(bbox: bounding_box::BoundingBox, confidence: Option<f32>, class_id: i32) -> Self {
        Self { bbox, confidence, class_id }
    }
}