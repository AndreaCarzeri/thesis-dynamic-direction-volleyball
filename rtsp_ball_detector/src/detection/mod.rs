pub mod bounding_box;
pub mod yolo;
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