use serde::{Deserialize, Serialize};

use opencv::{
    core::{self, Point, Rect, Scalar},
    imgproc,
    prelude::*,
    Result,
};

use super::Detection;

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct BoundingBox {
	pub x1: f32,
	pub y1: f32,
	pub x2: f32,
	pub y2: f32,
}

impl BoundingBox {
	pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
		Self { x1, y1, x2, y2 }
	}

	pub fn to_rect(&self) -> Rect {
		Rect::new(self.x1 as i32, self.y1 as i32, (self.x2 - self.x1) as i32, (self.y2 - self.y1) as i32)
	}

	// Function calculates "Intersection-over-union" coefficient for specified two boxes
	// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
	// Returns Intersection over union ratio as a float number
	pub fn iou(box1: &Self, box2: &Self) -> f32 {
		Self::intersection(box1, box2) / Self::union(box1, box2)
	}

	// Function calculates union area of two boxes
	// Returns Area of the boxes union as a float number
	fn union(box1: &Self, box2: &Self) -> f32 {
		let box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
		let box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
		box1_area + box2_area - Self::intersection(box1, box2)
	}

	// Function calculates intersection area of two boxes
	// Returns Area of intersection of the boxes as a float number
	fn intersection(box1: &Self, box2: &Self) -> f32 {
		let x1 = box1.x1.max(box2.x1);
		let y1 = box1.y1.max(box2.y1);
		let x2 = box1.x2.min(box2.x2);
		let y2 = box1.y2.min(box2.y2);
		(x2 - x1).max(0.0) * (y2 - y1).max(0.0)
	}

	// Add these as instance methods
	pub fn iou_with(&self, other: &Self) -> f32 {
		Self::iou(self, other)
	}

	fn union_with(&self, other: &Self) -> f32 {
		Self::union(self, other)
	}

	fn intersection_with(&self, other: &Self) -> f32 {
		Self::intersection(self, other)
	}
}

/// Extracts bounding boxes from a binary mask.
///
/// # Arguments
///
/// * `mask` - A binary mask where the objects are white (255) and the background is black (0).
///
/// # Returns
///
/// A vector of `Rect` structures representing the bounding boxes of detected objects.
pub fn extract_boxes(mask: &Mat) -> Result<Vec<Detection>> {
    let mut contours = core::Vector::<core::Vector<Point>>::new();
    imgproc::find_contours(
        mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::new(0, 0),
    )?;

	let boxes = contours
		.iter()
		.map(|contour| {
			let rect = imgproc::bounding_rect(&contour)?;
			Ok(Detection::new(
				BoundingBox::new(
					rect.x as f32,
					rect.y as f32,
					(rect.x + rect.width) as f32,
					(rect.y + rect.height) as f32,
				),
				None,
				1, // Default class_id
			))
		})
		.collect::<Result<Vec<Detection>>>()?;

    Ok(boxes)
}

/// Draws bounding boxes on an image.
///
/// # Arguments
///
/// * `image` - The image on which to draw the bounding boxes.
/// * `detections` - A vector of Detection structures representing the detection boxes to draw.
/// * `color` - The color of the bounding box edges.
/// * `thickness` - The thickness of the bounding box edges.
pub fn draw_boxes(image: &mut Mat, detections: &Vec<Detection>, color: Scalar, thickness: i32) -> Result<()> {
	let rects = detections.iter().map(|det| det.bbox.to_rect()).collect::<Vec<Rect>>();
    for rect in rects {
        imgproc::rectangle(image, rect, color, thickness, imgproc::LINE_8, 0)?;
    }
    Ok(())
}

pub fn non_maximum_suppression(
	detections: &mut Vec<Detection>,
	nms_threshold: f32,
) -> Vec<Detection> {
	if detections.is_empty() {
		return Vec::new();
	}

	// Sort detections by confidence in descending order
	detections.sort_by(|a, b| b.confidence.unwrap_or(0.0).partial_cmp(&a.confidence.unwrap_or(0.0)).unwrap());

	let mut result = Vec::new();
	let mut detections_clone = detections.clone();

	while !detections_clone.is_empty() {
		// Take the detection with the highest confidence
		let best_detection = detections_clone.remove(0);
		result.push(best_detection);

		// Filter out all other detections that have a high IoU with the best one
		detections_clone.retain(|det| {
			BoundingBox::iou(&best_detection.bbox, &det.bbox) < nms_threshold
		});
	}
	result
}
