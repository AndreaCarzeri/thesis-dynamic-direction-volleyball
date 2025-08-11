use super::bounding_box::BoundingBox;
use super::Detection;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use quick_xml::events::Event;
use quick_xml::Reader;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};

// Structure to hold frame detections
#[derive(Debug)]
pub struct FrameDetections {
    frame_id: usize,
    detections: Vec<Detection>,
}

// Structure to store evaluation results
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub iou: f32,
}

// Structure to store evaluation across multiple frames
#[derive(Debug)]
pub struct DetectionEvaluator {
    // Store ground truth and predictions for each frame
    frame_gt: HashMap<usize, Vec<Detection>>,
    frame_pred: HashMap<usize, Vec<Detection>>,
    
    // Configurable IoU threshold
    iou_threshold: f32,
    
    // For confidence-based evaluation
    with_confidence: bool,
}

impl DetectionEvaluator {
    pub fn new(iou_threshold: f32, with_confidence: bool) -> Self {
        Self {
            frame_gt: HashMap::new(),
            frame_pred: HashMap::new(),
            iou_threshold,
            with_confidence,
        }
    }
    
    // Add ground truth for a frame
    pub fn add_ground_truth(&mut self, frame_id: usize, detections: Vec<Detection>) {
        self.frame_gt.insert(frame_id, detections);
    }
    
    // Add predictions for a frame
    pub fn add_predictions(&mut self, frame_id: usize, detections: Vec<Detection>) {
        self.frame_pred.insert(frame_id, detections);
    }
    
    // Calculate IoU between two detections
    fn calc_iou(&self, det1: &Detection, det2: &Detection) -> f32 {
        det1.bbox.iou_with(&det2.bbox)
    }
    
    // Evaluate a single frame
    pub fn evaluate_frame(&self, frame_id: usize) -> Option<EvaluationMetrics> {
        let gt = self.frame_gt.get(&frame_id)?;
        let pred = self.frame_pred.get(&frame_id)?;
        
        if gt.is_empty() && pred.is_empty() {
            return None;
        }
        
        let mut tp = 0;
        let mut fp = 0;
        let mut total_iou = 0.0;
        let mut matched_gt_indices = vec![false; gt.len()];
        
        // Sort predictions by confidence if using confidence
        let mut sorted_pred = pred.clone();
        if self.with_confidence {
            sorted_pred.sort_by(|a, b| {
                b.confidence.unwrap_or(0.0).partial_cmp(&a.confidence.unwrap_or(0.0)).unwrap()
            });
        }
        
        // Match predictions to ground truth
        for pred_det in &sorted_pred {
            let mut best_iou = self.iou_threshold;
            let mut best_gt_idx = None;
            
            // Find best matching ground truth
            for (gt_idx, gt_det) in gt.iter().enumerate() {
                if matched_gt_indices[gt_idx] {
                    continue;
                }
                
                let iou = self.calc_iou(pred_det, gt_det);
                if iou > best_iou {
                    best_iou = iou;
                    best_gt_idx = Some(gt_idx);
                }
            }
            
            // If match found
            if let Some(gt_idx) = best_gt_idx {
                tp += 1;
                matched_gt_indices[gt_idx] = true;
                total_iou += best_iou;
            } else {
                fp += 1;
            }
        }
        
        let fn_count = gt.len() - tp;
        
        // Calculate metrics
        let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 
            2.0 * precision * recall / (precision + recall) 
        } else { 
            0.0 
        };
        let avg_iou = if tp > 0 { total_iou / tp as f32 } else { 0.0 };
        
        Some(EvaluationMetrics {
            precision,
            recall,
            f1_score,
            iou: avg_iou,
        })
    }
    
    // Evaluate all frames and calculate average metrics
    pub fn evaluate_all_frames(&self) -> EvaluationMetrics {
        let mut total_metrics = EvaluationMetrics {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            iou: 0.0,
        };
        
        let mut frame_count = 0;
        
        // Process each frame that has both ground truth and predictions
        for frame_id in self.frame_gt.keys() {
            if let Some(metrics) = self.evaluate_frame(*frame_id) {
                total_metrics.precision += metrics.precision;
                total_metrics.recall += metrics.recall;
                total_metrics.f1_score += metrics.f1_score;
                total_metrics.iou += metrics.iou;
                frame_count += 1;
            }
        }
        
        // Calculate averages
        if frame_count > 0 {
            total_metrics.precision /= frame_count as f32;
            total_metrics.recall /= frame_count as f32;
            total_metrics.f1_score /= frame_count as f32;
            total_metrics.iou /= frame_count as f32;
        }
        
        total_metrics
    }
    
    // Calculate mAP (mean Average Precision)
    pub fn calculate_map(&self) -> f32 {
        if !self.with_confidence {
            println!("Warning: mAP calculation requires confidence scores");
            return 0.0;
        }
        
        let recall_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut average_precision = 0.0;
        
        // Collect all predictions and ground truths across frames
        let mut all_predictions: Vec<(Detection, usize)> = Vec::new();
        let mut all_gt: HashMap<usize, Vec<Detection>> = HashMap::new();
        
        for (frame_id, preds) in &self.frame_pred {
            for pred in preds {
                all_predictions.push((*pred, *frame_id));
            }
        }
        
        all_gt = self.frame_gt.clone();
        
        // Sort by confidence
        all_predictions.sort_by(|a, b| {
            b.0.confidence.unwrap_or(0.0).partial_cmp(&a.0.confidence.unwrap_or(0.0)).unwrap()
        });
        
        // Calculate precision at different recall levels
        let mut matches = Vec::new();
        let mut gt_matched: HashMap<usize, Vec<bool>> = HashMap::new();
        
        for (frame_id, gt_dets) in &all_gt {
            gt_matched.insert(*frame_id, vec![false; gt_dets.len()]);
        }
        
        let total_gt: usize = all_gt.values().map(|v| v.len()).sum();
        
        // Process predictions in order of confidence
        for (pred, frame_id) in &all_predictions {
            if let Some(gt_dets) = all_gt.get(frame_id) {
                let mut best_iou = self.iou_threshold;
                let mut best_gt_idx = None;
                
                // Find best matching ground truth for this prediction
                for (gt_idx, gt_det) in gt_dets.iter().enumerate() {
                    if gt_matched[frame_id][gt_idx] {
                        continue;
                    }
                    
                    let iou = self.calc_iou(pred, gt_det);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gt_idx = Some(gt_idx);
                    }
                }
                
                if let Some(gt_idx) = best_gt_idx {
                    matches.push(true);
                    gt_matched.get_mut(frame_id).unwrap()[gt_idx] = true;
                } else {
                    matches.push(false);
                }
            } else {
                matches.push(false);
            }
        }
        
        // Calculate precision and recall at each prediction
        let mut precisions = Vec::new();
        let mut recalls = Vec::new();
        
        let mut true_positives = 0;
        for (i, is_match) in matches.iter().enumerate() {
            if *is_match {
                true_positives += 1;
            }
            
            let precision = true_positives as f32 / (i + 1) as f32;
            let recall = if total_gt > 0 {
                true_positives as f32 / total_gt as f32
            } else {
                0.0
            };
            
            precisions.push(precision);
            recalls.push(recall);
        }
        
        // Calculate average precision at specified recall thresholds
        for i in 0..(recall_thresholds.len() - 1) {
            let start_recall = recall_thresholds[i];
            let end_recall = recall_thresholds[i + 1];
            let mut max_precision = 0.0f32;
            
            // Find maximum precision for recalls in this range
            for (j, recall) in recalls.iter().enumerate() {
                if *recall >= start_recall && *recall <= end_recall {
                    max_precision = max_precision.max(precisions[j]);
                }
            }
            
            average_precision += max_precision * 0.1; // Each recall range is 0.1 wide
        }
        
        average_precision
    }
}

// Example usage function
pub fn evaluate_volleyball_detections(
    ground_truth: HashMap<usize, Vec<Detection>>,
    predictions: HashMap<usize, Vec<Detection>>,
    iou_threshold: f32,
    with_confidence: bool
) -> (EvaluationMetrics, f32) {
    let mut evaluator = DetectionEvaluator::new(iou_threshold, with_confidence);
    
    // Add all ground truth and predictions
    for (frame_id, gt_dets) in ground_truth {
        evaluator.add_ground_truth(frame_id, gt_dets);
    }
    
    for (frame_id, pred_dets) in predictions {
        evaluator.add_predictions(frame_id, pred_dets);
    }
    
    // Calculate metrics
    let metrics = evaluator.evaluate_all_frames();
    let map = evaluator.calculate_map();
    
    (metrics, map)
}

// Directly parse XML into the format needed for evaluation
pub fn parse_cvat_xml_for_evaluation(
    file_path: &str, 
    max_frames_per_track: Option<usize>
) -> Result<HashMap<usize, Vec<Detection>>> {
    let path = Path::new(file_path);
    let file = File::open(path)
        .with_context(|| format!("Failed to open XML file: {}", file_path))?;
    let file = BufReader::new(file);
    
    let mut reader = Reader::from_reader(file);
    
    let mut frame_detections: HashMap<usize, Vec<Detection>> = HashMap::new();
    let mut buf = Vec::new();
    
    let mut in_track = false;
    let mut current_track_id = 0;
    // Parse XML
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"track" => {
                        in_track = true;
                    },
                    b"box" if in_track => {
						let mut frame = 0;
                        let mut outside = false;
                        let mut x1 = 0.0;
                        let mut y1 = 0.0;
                        let mut x2 = 0.0;
                        let mut y2 = 0.0;
                        
                        // Extract box attributes
                        for attr in e.attributes().filter_map(Result::ok) {
                            if let Ok(value) = std::str::from_utf8(&attr.value) {
                                match attr.key.local_name().as_ref() {
                                    b"frame" => frame = value.parse::<usize>()
										.with_context(|| format!("Failed to parse frame number: {}", value))?,
                                    b"outside" => outside = value == "1",
                                    b"xtl" => x1 = value.parse::<f32>()
										.with_context(|| format!("Failed to parse x1 coordinate: {}", value))?,
                                    b"ytl" => y1 = value.parse::<f32>()
										.with_context(|| format!("Failed to parse y1 coordinate: {}", value))?,
                                    b"xbr" => x2 = value.parse::<f32>()
										.with_context(|| format!("Failed to parse x2 coordinate: {}", value))?,
                                    b"ybr" => y2 = value.parse::<f32>()
										.with_context(|| format!("Failed to parse y2 coordinate: {}", value))?,
                                    _ => {}
                                }
                            }
                        }
                        
                        // Only include boxes with outside="0" and respect max_frames_per_track
                        if !outside {
                            if max_frames_per_track.is_none() || frame <= max_frames_per_track.unwrap() {
                                // Create BoundingBox and Detection objects
                                let bbox = BoundingBox::new(x1, y1, x2, y2);
								// current_track_id could be player id for future implementation
                                let detection = Detection::new(bbox, None, 1);
                                
                                // Add to frame_detections HashMap
                                frame_detections
                                    .entry(frame)
                                    .or_insert_with(Vec::new)
                                    .push(detection);
                            }
                        }
                    },
                    _ => {}
                }
            },
            Ok(Event::End(ref e)) => {
                if e.name() == quick_xml::name::QName(b"track") {
                    in_track = false;
                }
            },
            Ok(Event::Eof) => break,
            Err(e) => anyhow::bail!(format!("Error parsing XML: {}", e)),
            _ => {}
        }
        buf.clear();
    }
    
    Ok(frame_detections)
}

/// Saves predictions to a JSON file
pub fn save_predictions(
    predictions: &HashMap<usize, Vec<Detection>>,
    filename: &str,
) -> Result<()> {
    let json = serde_json::to_string_pretty(predictions)
        .context("Failed to serialize predictions to JSON")?;
    
    let mut file = File::create(filename)
        .with_context(|| format!("Failed to create file: {}", filename))?;
    
    file.write_all(json.as_bytes())
        .with_context(|| format!("Failed to write data to file: {}", filename))?;
    
    Ok(())
}

/// Loads predictions from a JSON file
pub fn load_predictions(filename: &str) -> Result<HashMap<usize, Vec<Detection>>> {
    let path = Path::new(filename);
    
    // Check if file exists
    if !path.exists() {
        anyhow::bail!("File not found: {}", filename);
    }
    
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", filename))?;
    
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .with_context(|| format!("Failed to read contents of file: {}", filename))?;
    
    let predictions: HashMap<usize, Vec<Detection>> = serde_json::from_str(&contents)
        .context("Failed to parse JSON into predictions")?;
    
    Ok(predictions)
}