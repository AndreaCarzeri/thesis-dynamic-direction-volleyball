use serde::{Deserialize, Serialize};
use opencv::{
	core::{Rect, Scalar},
	imgproc,
	prelude::*,
	Result,
};
use super::Detection;

/// Represents a bounding box with coordinates.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct BoundingBox {
	pub x1: f32,
	pub y1: f32,
	pub x2: f32,
	pub y2: f32,
}

impl BoundingBox {
	/// Creates a new bounding box.
	pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
		Self { x1, y1, x2, y2 }
	}

	/// Converts the bounding box to an OpenCV `Rect`.
	pub fn to_rect(self) -> Rect {
		Rect::new(self.x1 as i32, self.y1 as i32, (self.x2 - self.x1) as i32, (self.y2 - self.y1) as i32)
	}

	/// Calculates the Intersection-over-Union (IoU) of two boxes.
	pub fn iou(box1: &Self, box2: &Self) -> f32 {
		Self::intersection(box1, box2) / Self::union(box1, box2)
	}

	/// Calculates the union area of two boxes.
	fn union(box1: &Self, box2: &Self) -> f32 {
		let box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
		let box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
		box1_area + box2_area - Self::intersection(box1, box2)
	}

	/// Calculates the intersection area of two boxes.
	fn intersection(box1: &Self, box2: &Self) -> f32 {
		let x1 = box1.x1.max(box2.x1);
		let y1 = box1.y1.max(box2.y1);
		let x2 = box1.x2.min(box2.x2);
		let y2 = box1.y2.min(box2.y2);
		(x2 - x1).max(0.0) * (y2 - y1).max(0.0)
	}
}

/// Draws bounding boxes on a given image.
pub fn draw_boxes(image: &mut Mat, detections: &[Detection], color: Scalar, thickness: i32) -> Result<()> {
	let rects = detections.iter().map(|det| det.bbox.to_rect()).collect::<Vec<Rect>>();
	for rect in rects {
		imgproc::rectangle(image, rect, color, thickness, imgproc::LINE_8, 0)?;
	}
	Ok(())
}

/// Applies Non-Maximum Suppression to filter overlapping detections.
pub fn non_maximum_suppression(
	detections: &mut Vec<Detection>,
	nms_threshold: f32,
) -> Vec<Detection> {
	if detections.is_empty() {
		return Vec::new();
	}

	detections.sort_by(|a, b| b.confidence.unwrap_or(0.0).partial_cmp(&a.confidence.unwrap_or(0.0)).unwrap());

	let mut result = Vec::new();
	let mut detections_clone = detections.to_owned();

	while !detections_clone.is_empty() {
		let best_detection = detections_clone.remove(0);
		result.push(best_detection);

		detections_clone.retain(|det| {
			BoundingBox::iou(&best_detection.bbox, &det.bbox) < nms_threshold
		});
	}
	result
}