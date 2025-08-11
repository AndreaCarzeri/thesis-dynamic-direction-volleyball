use std::collections::HashMap;
use anyhow::Result;
use itertools::Itertools;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use super::background_subtractor::BackgroundSubtractor;
use super::Detection;
use super::evaluation::EvaluationMetrics;

#[derive(Debug, Clone, Copy)]
struct ModelParams {
    confidence_threshold: f32,
    nms_threshold: f32,
    input_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ModelType {
    MOG2(MOG2Params),
    KNN(KNNParams),
    YOLO(YOLOParams),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct MOG2Params {
    var_threshold: f64, 
    n_mixtures: i32,
	iou_threshold: f32,
}

impl Eq for MOG2Params {}

impl std::hash::Hash for MOG2Params {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.var_threshold.to_bits().hash(state);
        self.n_mixtures.hash(state);
		self.iou_threshold.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct KNNParams {
    dist2_threshold: f64, 
    knn_samples: i32,
	iou_threshold: f32,
}

impl Eq for KNNParams {}

impl std::hash::Hash for KNNParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dist2_threshold.to_bits().hash(state);
        self.knn_samples.hash(state);
		self.iou_threshold.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct YOLOParams {
    nms_threshold: f32,
    confidence_threshold: f32,
	iou_threshold: f32,
}

impl Eq for YOLOParams {}

impl std::hash::Hash for YOLOParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nms_threshold.to_bits().hash(state);
        self.confidence_threshold.to_bits().hash(state);
		self.iou_threshold.to_bits().hash(state);
    }
}

pub fn grid_search_KNN(dataset: &str) -> Result<HashMap<ModelType, (EvaluationMetrics, f32)>> {
    // Define parameter ranges to test
    let dist2_thresholds = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
	let knn_samples = vec![2, 3, 4, 5];
	let ious = vec![0.0, 0.1, 0.3, 0.5];
	
	// Load ground truth detections
	let gt = super::evaluation::parse_cvat_xml_for_evaluation(
		"../assets/result.xml", 
		Some(105)
	)?;
	let frame_count = gt.keys().max().unwrap_or(&0);
	println!("Frame count: {frame_count}");
	
	if frame_count == &0 {
		anyhow::bail!("No ground truth detections found");
	}

    // Generate all combinations
    let parameter_combinations: Vec<KNNParams> = dist2_thresholds.iter()
        .cartesian_product(knn_samples.iter())
		.cartesian_product(ious.iter())
        .map(|((dist2_threshold, k), iou)| KNNParams {
			dist2_threshold: *dist2_threshold,
			knn_samples: *k,
			iou_threshold: *iou,
		})
        .collect();
    
	let tot = parameter_combinations.len();
    let mut current = 1;
	println!("Testing {} parameter combinations for KNN...", tot);

    // Evaluate each combination in parallel
    let results: HashMap<ModelType, (EvaluationMetrics, f32)> = parameter_combinations.iter()
		//.par_iter()
        .map(|params| {
			println!("{current}/{tot} Testing KNN with parameters: {:?}", params);
			let mut knn = super::background_subtractor::KNN::new_with_par(params.dist2_threshold, params.knn_samples).expect("Failed to create KNN model");
			let detections = knn.detect_video(dataset, *frame_count).expect("Failed to detect with KNN");
			let metrics = super::evaluation::evaluate_volleyball_detections(gt.clone(), detections, params.iou_threshold, false);
			println!("Metrics: {:?}", metrics);
			current += 1;
			(ModelType::KNN(*params), metrics)
        })
        .collect();
    
    Ok(results)
}

pub fn grid_search_MOG2(dataset: &str) -> Result<HashMap<ModelType, (EvaluationMetrics, f32)>> {
    // Define parameter ranges to test
    let var_thresholds = vec![110.0, 130.0, 150.0, 170.0, 190.0];
    let n_mixtures = vec![1, 2, 3, 4];
    let ious = vec![0.0, 0.1, 0.3, 0.5];
    
    // Load ground truth detections
    let gt = super::evaluation::parse_cvat_xml_for_evaluation(
		"../assets/result.xml", 
		Some(105)
	)?;
	let frame_count = gt.keys().max().unwrap_or(&0);

    // Generate all combinations
    let parameter_combinations: Vec<MOG2Params> = var_thresholds.iter()
        .cartesian_product(n_mixtures.iter())
        .cartesian_product(ious.iter())
        .map(|((var_threshold, n), iou)| MOG2Params {
            var_threshold: *var_threshold,
            n_mixtures: *n,
            iou_threshold: *iou,
        })
        .collect();
    
	let tot = parameter_combinations.len();
    println!("Testing {} parameter combinations for MOG2...", tot);
	let mut current = 1;
    
    // Evaluate each combination in parallel
    let results: HashMap<ModelType, (EvaluationMetrics, f32)> = parameter_combinations.iter()//.par_iter()
        .map(|params| {
			println!("{current}/{tot} Testing MOG2 with parameters: {:?}", params);
            let mut mog2 = super::background_subtractor::MOG2::new_with_par(params.var_threshold, params.n_mixtures).expect("Failed to create MOG2 model");
    		let detections = mog2.detect_video(dataset, *frame_count).expect("Failed to detect with MOG2");
    		let metrics = super::evaluation::evaluate_volleyball_detections(gt.clone(), detections, params.iou_threshold, false);
			println!("Metrics: {:?}", metrics);
			current += 1;
			(ModelType::MOG2(*params), metrics)
        })
        .collect();
    
    Ok(results)
}

pub fn grid_search_YOLO(dataset: &str) -> Result<HashMap<ModelType, (EvaluationMetrics, f32)>> {
    // Define parameter ranges to test
    let confidence_thresholds = vec![0.4, 0.45, 0.5, 0.55, 0.6];
    let nms_thresholds = vec![0.4, 0.45, 0.5, 0.55, 0.6];
    let ious = vec![0.0, 0.1, 0.3, 0.5];
    
    // Load ground truth detections
    let gt = super::evaluation::parse_cvat_xml_for_evaluation(
		"../assets/result.xml", 
		Some(105)
	)?;
	let frame_count = gt.keys().max().unwrap_or(&0);

    // Generate all combinations
    let parameter_combinations: Vec<YOLOParams> = confidence_thresholds.iter()
        .cartesian_product(nms_thresholds.iter())
        .cartesian_product(ious.iter())
        .map(|((conf, nms), iou)| YOLOParams {
            confidence_threshold: *conf,
            nms_threshold: *nms,
            iou_threshold: *iou,
        })
        .collect();
    
	let tot = parameter_combinations.len();
    println!("Testing {} parameter combinations for YOLO...", parameter_combinations.len());
	let mut current = 1;
    
    // Evaluate each combination in parallel
    let results: HashMap<ModelType, (EvaluationMetrics, f32)> = parameter_combinations.iter()//.par_iter()
        .map(|params| {
			println!("{current}/{tot} Testing YOLO with parameters: {:?}", params);
            let mut yolo = super::yolo::YoloDetector::new("../assets/best2.onnx", params.confidence_threshold, params.nms_threshold).expect("Failed to create YOLO model");
            let detections = yolo.detect_video(dataset, *frame_count).expect("Failed to detect with YOLO");
    		let metrics = super::evaluation::evaluate_volleyball_detections(gt.clone(), detections, params.iou_threshold, true);
			println!("Metrics: {:?}", metrics);
			current += 1;
			(ModelType::YOLO(*params), metrics)
        })
        .collect();
    
    Ok(results)
}

fn get_model_name_and_params(model_type: &ModelType) -> (String, Vec<(String, String)>) {
    match model_type {
        ModelType::KNN(params) => (
            "KNN".to_string(),
            vec![
                ("dist2_threshold".to_string(), format!("{:.1}", params.dist2_threshold)),
                ("knn_samples".to_string(), format!("{}", params.knn_samples)),
                ("iou_threshold".to_string(), format!("{:.1}", params.iou_threshold)),
            ]
        ),
        ModelType::MOG2(params) => (
            "MOG2".to_string(),
            vec![
                ("var_threshold".to_string(), format!("{:.1}", params.var_threshold)),
                ("n_mixtures".to_string(), format!("{}", params.n_mixtures)),
                ("iou_threshold".to_string(), format!("{:.1}", params.iou_threshold)),
            ]
        ),
        ModelType::YOLO(params) => (
            "YOLO".to_string(),
            vec![
                ("conf_threshold".to_string(), format!("{:.2}", params.confidence_threshold)),
                ("nms_threshold".to_string(), format!("{:.2}", params.nms_threshold)),
                ("iou_threshold".to_string(), format!("{:.1}", params.iou_threshold)),
            ]
        ),
    }
}

pub fn plot_evaluation_results_by_model_type(results: &HashMap<ModelType, (EvaluationMetrics, f32)>, model_type_filter: Option<&str>, output_path: &str) -> Result<()> {
    // Filter results by model type if specified
    let filtered_results: HashMap<ModelType, (EvaluationMetrics, f32)> = match model_type_filter {
        Some(filter) => results.iter()
            .filter(|(k, _)| match k {
                ModelType::KNN(_) if filter == "KNN" => true,
                ModelType::MOG2(_) if filter == "MOG2" => true,
                ModelType::YOLO(_) if filter == "YOLO" => true,
                _ => false,
            })
            .map(|(k, v)| (*k, *v))
            .collect(),
        None => results.clone(),
    };
    
    if filtered_results.is_empty() {
        println!("No results to plot after filtering");
        return Ok(());
    }
    
    // Extract the best parameters based on F1 score
    let best_params = filtered_results.iter()
        .max_by(|a, b| {
            a.1.0.f1_score.partial_cmp(&b.1.0.f1_score).unwrap()
        })
        .map(|(k, v)| (*k, *v))
        .unwrap();
    
    // Create a bitmap to draw on
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);
    
    // Split into two charts
    let (upper, lower) = root.split_vertically(400);
    
    // Draw model-specific parameter analysis
    match best_params.0 {
        ModelType::KNN(_) => plot_knn_analysis(&filtered_results, &upper, &lower)?,
        ModelType::MOG2(_) => plot_mog2_analysis(&filtered_results, &upper, &lower)?,
        ModelType::YOLO(_) => plot_yolo_analysis(&filtered_results, &upper, &lower)?,
    };
    
    // Display the best parameters
    let (model_name, params) = get_model_name_and_params(&best_params.0);
    println!("Best parameters for {}", model_name);
    for (name, value) in params {
        println!("  {}: {}", name, value);
    }
    println!("Best metrics:");
    println!("  Precision: {:.4}", best_params.1.0.precision);
    println!("  Recall: {:.4}", best_params.1.0.recall);
    println!("  F1 Score: {:.4}", best_params.1.0.f1_score);
    println!("  Average IoU: {:.4}", best_params.1.0.iou);
    println!("  mAP: {:.4}", best_params.1.1);
    
    Ok(())
}

fn plot_knn_analysis(
    results: &HashMap<ModelType, (EvaluationMetrics, f32)>, 
    upper: &DrawingArea<BitMapBackend, plotters::coord::Shift>, 
    lower: &DrawingArea<BitMapBackend, plotters::coord::Shift>
) -> Result<()> {
    // Extract KNN params and metrics
    let knn_results: Vec<(KNNParams, EvaluationMetrics, f32)> = results.iter()
        .filter_map(|(model_type, metrics)| {
            if let ModelType::KNN(params) = model_type {
                Some((*params, metrics.0, metrics.1))
            } else {
                None
            }
        })
        .collect();
    
	// Get unique values
	let dist2_thresholds: Vec<f64> = knn_results.iter()
		.map(|(p, _, _)| p.dist2_threshold)
		.collect::<Vec<f64>>()  // First collect to a Vec
		.into_iter()
		.sorted_by(|a, b| a.partial_cmp(b).unwrap())  // Sort them
		.dedup()  // Remove duplicates after sorting (requires itertools)
		.collect();

	let knn_samples_values: Vec<i32> = knn_results.iter()
		.map(|(p, _, _)| p.knn_samples)
		.collect::<std::collections::HashSet<i32>>()
		.into_iter()
		.sorted()
		.collect();

    // Create line chart for dist2_threshold vs metrics (fixed best knn_samples and iou)
    {
        // Find the best parameters
        let best_params = knn_results.iter()
            .max_by(|a, b| a.1.f1_score.partial_cmp(&b.1.f1_score).unwrap())
            .unwrap();
        
        let best_knn_samples = best_params.0.knn_samples;
        let best_iou = best_params.0.iou_threshold;
        
        // Filter for the best knn_samples and iou
        let threshold_results: Vec<(f64, &EvaluationMetrics)> = knn_results.iter()
            .filter(|(params, _, _)| params.knn_samples == best_knn_samples && 
                                    (params.iou_threshold - best_iou).abs() < 0.01)
            .map(|(params, metrics, _)| (params.dist2_threshold, metrics))
            .sorted_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .collect();
        
        let mut chart = ChartBuilder::on(upper)
            .caption(format!("KNN Metrics vs Dist² Threshold (K={}, IoU={:.1})", best_knn_samples, best_iou), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d((dist2_thresholds[0] - 50.0)..(dist2_thresholds[dist2_thresholds.len() - 1] + 50.0), 0f32..1f32)?;
        
        chart.configure_mesh()
            .x_desc("Distance² Threshold")
            .y_desc("Metric Value")
            .draw()?;
        
        // Extract series
        let precision_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.precision))
            .collect();
        
        let recall_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.recall))
            .collect();
        
        let f1_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.f1_score))
            .collect();
        
        let iou_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.iou))
            .collect();
        
        // Plot series
        chart.draw_series(LineSeries::new(precision_series, &RED))?
            .label("Precision")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        
        chart.draw_series(LineSeries::new(recall_series, &BLUE))?
            .label("Recall")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
        
        chart.draw_series(LineSeries::new(f1_series, &GREEN))?
            .label("F1 Score")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        
        chart.draw_series(LineSeries::new(iou_series, &MAGENTA))?
            .label("Avg IoU")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }
    
    // Create heat map for F1 scores across dist2_threshold and knn_samples
    {   
        // Find the best IOU threshold
        let best_iou = knn_results.iter()
            .max_by(|a, b| a.1.f1_score.partial_cmp(&b.1.f1_score).unwrap())
            .map(|(p, _, _)| p.iou_threshold)
            .unwrap();
        
        let mut chart = ChartBuilder::on(lower)
            .caption(format!("KNN F1 Score Heatmap (IoU Threshold = {:.1})", best_iou), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (dist2_thresholds[0] - 25.0)..(dist2_thresholds[dist2_thresholds.len() - 1] + 25.0),
                (knn_samples_values[0] as f64 - 0.5)..(knn_samples_values[knn_samples_values.len() - 1] as f64 + 0.5)
            )?;
        
        chart.configure_mesh()
            .x_desc("Distance² Threshold")
            .y_desc("KNN Samples (K)")
            .disable_mesh()
            .draw()?;
        
        // Create heatmap data
        let cell_width = (dist2_thresholds[1] - dist2_thresholds[0]) * 0.8;
        let cell_height = 0.8;
        
        let mut heatmap_data = vec![];
        for &dist2 in &dist2_thresholds {
            for &k in &knn_samples_values {
                if let Some(entry) = knn_results.iter().find(|(p, _, _)| 
                    (p.dist2_threshold - dist2).abs() < 0.01 && 
                    p.knn_samples == k && 
                    (p.iou_threshold - best_iou).abs() < 0.01
                ) {
                    heatmap_data.push((dist2, k as f64, entry.1.f1_score as f64));
                }
            }
        }
        
        chart.draw_series(
            heatmap_data.iter().map(|&(x, y, f1)| {
                let color = HSLColor(
                    0.6 - f1 * 0.6, // Blue to red (hue)
                    0.8,            // saturation
                    0.3 + f1 * 0.4  // lightness
                );
                Rectangle::new(
                    [(x - cell_width/2.0, y - cell_height/2.0), (x + cell_width/2.0, y + cell_height/2.0)],
                    color.filled(),
                )
            })
        )?;
        
        // Add F1 score labels
        chart.draw_series(
            heatmap_data.iter().map(|&(x, y, f1)| {
                Text::new(
                    format!("{:.2}", f1),
                    (x, y),
                    ("sans-serif", 15.0).into_font().color(&WHITE),
                )
            })
        )?;
    }
    
    Ok(())
}

fn plot_mog2_analysis(
    results: &HashMap<ModelType, (EvaluationMetrics, f32)>, 
    upper: &DrawingArea<BitMapBackend, plotters::coord::Shift>, 
    lower: &DrawingArea<BitMapBackend, plotters::coord::Shift>
) -> Result<()> {
    // Extract MOG2 params and metrics
    let mog2_results: Vec<(MOG2Params, EvaluationMetrics, f32)> = results.iter()
        .filter_map(|(model_type, metrics)| {
            if let ModelType::MOG2(params) = model_type {
                Some((*params, metrics.0, metrics.1))
            } else {
                None
            }
        })
        .collect();
    
	// Get unique values - using proper approach for f64 values
	let var_thresholds: Vec<f64> = mog2_results.iter()
		.map(|(p, _, _)| p.var_threshold)
		.collect::<Vec<f64>>()
		.into_iter()
		.sorted_by(|a, b| a.partial_cmp(b).unwrap())
		.dedup()
		.collect();

	let n_mixtures_values: Vec<i32> = mog2_results.iter()
		.map(|(p, _, _)| p.n_mixtures)
		.collect::<std::collections::HashSet<i32>>()
		.into_iter()
		.sorted()
		.collect();

    // Create line chart for var_threshold vs metrics (fixed best n_mixtures and iou)
    {
        // Find the best parameters
        let best_params = mog2_results.iter()
            .max_by(|a, b| a.1.f1_score.partial_cmp(&b.1.f1_score).unwrap())
            .unwrap();
        
        let best_n_mixtures = best_params.0.n_mixtures;
        let best_iou = best_params.0.iou_threshold;
        
        // Filter for the best n_mixtures and iou
        let threshold_results: Vec<(f64, &EvaluationMetrics)> = mog2_results.iter()
            .filter(|(params, _, _)| params.n_mixtures == best_n_mixtures && 
                                    (params.iou_threshold - best_iou).abs() < 0.01)
            .map(|(params, metrics, _)| (params.var_threshold, metrics))
            .sorted_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .collect();
        
        let mut chart = ChartBuilder::on(upper)
            .caption(format!("MOG2 Metrics vs Var Threshold (N={}, IoU={:.1})", best_n_mixtures, best_iou), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d((var_thresholds[0] - 5.0)..(var_thresholds[var_thresholds.len() - 1] + 5.0), 0f32..1f32)?;
        
        chart.configure_mesh()
            .x_desc("Variance Threshold")
            .y_desc("Metric Value")
            .draw()?;
        
        // Extract series
        let precision_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.precision))
            .collect();
        
        let recall_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.recall))
            .collect();
        
        let f1_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.f1_score))
            .collect();
        
        let iou_series: Vec<(f64, f32)> = threshold_results.iter()
            .map(|(threshold, metrics)| (*threshold, metrics.iou))
            .collect();
        
        // Plot series
        chart.draw_series(LineSeries::new(precision_series, &RED))?
            .label("Precision")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        
        chart.draw_series(LineSeries::new(recall_series, &BLUE))?
            .label("Recall")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
        
        chart.draw_series(LineSeries::new(f1_series, &GREEN))?
            .label("F1 Score")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        
        chart.draw_series(LineSeries::new(iou_series, &MAGENTA))?
            .label("Avg IoU")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }
    
    // Create heat map for F1 scores across var_threshold and n_mixtures
    {        
        // Find the best IOU threshold
        let best_iou = mog2_results.iter()
            .max_by(|a, b| a.1.f1_score.partial_cmp(&b.1.f1_score).unwrap())
            .map(|(p, _, _)| p.iou_threshold)
            .unwrap();
        
        let mut chart = ChartBuilder::on(lower)
            .caption(format!("MOG2 F1 Score Heatmap (IoU Threshold = {:.1})", best_iou), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (var_thresholds[0] - 5.0)..(var_thresholds[var_thresholds.len() - 1] + 5.0),
                (n_mixtures_values[0] as f64 - 0.5)..(n_mixtures_values[n_mixtures_values.len() - 1] as f64 + 0.5)
            )?;
        
        chart.configure_mesh()
            .x_desc("Variance Threshold")
            .y_desc("Number of Mixtures")
            .disable_mesh()
            .draw()?;
        
        // Create heatmap data
        let cell_width = (var_thresholds[1] as f64 - var_thresholds[0] as f64) * 0.8;
        let cell_height = 0.8;
        
        let mut heatmap_data = vec![];
        for &var in &var_thresholds {
            for &n in &n_mixtures_values {
                if let Some(entry) = mog2_results.iter().find(|(p, _, _)| 
                    (p.var_threshold - var).abs() < 0.01 && 
                    p.n_mixtures == n && 
                    (p.iou_threshold - best_iou).abs() < 0.01
                ) {
                    heatmap_data.push((var, n as f64, entry.1.f1_score as f64));
                }
            }
        }
        
        chart.draw_series(
            heatmap_data.iter().map(|&(x, y, f1)| {
                let color = HSLColor(
                    0.6 - f1 * 0.6, // Blue to red
                    0.8,
                    0.3 + f1 * 0.4,
                );
                Rectangle::new(
                    [(x - cell_width/2.0, y - cell_height/2.0), (x + cell_width/2.0, y + cell_height/2.0)],
                    color.filled(),
                )
            })
        )?;
        
        // Add F1 score labels
        chart.draw_series(
            heatmap_data.iter().map(|&(x, y, f1)| {
                Text::new(
                    format!("{:.2}", f1),
                    (x, y),
                    ("sans-serif", 15.0).into_font().color(&WHITE),
                )
            })
        )?;
    }
    
    Ok(())
}

fn plot_yolo_analysis(
    results: &HashMap<ModelType, (EvaluationMetrics, f32)>, 
    upper: &DrawingArea<BitMapBackend, plotters::coord::Shift>, 
    lower: &DrawingArea<BitMapBackend, plotters::coord::Shift>
) -> Result<()> {
    // Extract YOLO params and metrics
    let yolo_results: Vec<(YOLOParams, EvaluationMetrics, f32)> = results.iter()
        .filter_map(|(model_type, metrics)| {
            if let ModelType::YOLO(params) = model_type {
                Some((*params, metrics.0, metrics.1))
            } else {
                None
            }
        })
        .collect();

	// Get unique values - using proper approach for f32 values
	let conf_thresholds: Vec<f32> = yolo_results.iter()
		.map(|(p, _, _)| p.confidence_threshold)
		.collect::<Vec<f32>>()
		.into_iter()
		.sorted_by(|a, b| a.partial_cmp(b).unwrap())
		.dedup()
		.collect();

	let nms_thresholds: Vec<f32> = yolo_results.iter()
		.map(|(p, _, _)| p.nms_threshold)
		.collect::<Vec<f32>>()
		.into_iter()
		.sorted_by(|a, b| a.partial_cmp(b).unwrap())
		.dedup()
		.collect();
    
    // Create line chart for confidence_threshold vs metrics (fixed best nms_threshold and iou)
    {
        // Find the best parameters
        let best_params = yolo_results.iter()
            .max_by(|a, b| a.1.f1_score.partial_cmp(&b.1.f1_score).unwrap())
            .unwrap();
        
        let best_nms = best_params.0.nms_threshold;
        let best_iou = best_params.0.iou_threshold;
        
        // Filter for the best nms_threshold and iou
        let conf_results: Vec<(f32, &EvaluationMetrics)> = yolo_results.iter()
            .filter(|(params, _, _)| (params.nms_threshold - best_nms).abs() < 0.01 && 
                                    (params.iou_threshold - best_iou).abs() < 0.01)
            .map(|(params, metrics, _)| (params.confidence_threshold, metrics))
            .sorted_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .collect();
        
        let mut chart = ChartBuilder::on(upper)
            .caption(format!("YOLO Metrics vs Confidence Threshold (NMS={:.2}, IoU={:.1})", best_nms, best_iou), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d((conf_thresholds[0] - 0.25)..(conf_thresholds[conf_thresholds.len() - 1] + 0.25), 0f32..1f32)?;
        
        chart.configure_mesh()
            .x_desc("Confidence Threshold")
            .y_desc("Metric Value")
            .draw()?;
        
        // Extract series
        let precision_series: Vec<(f32, f32)> = conf_results.iter()
            .map(|(conf, metrics)| (*conf, metrics.precision))
            .collect();
        
        let recall_series: Vec<(f32, f32)> = conf_results.iter()
            .map(|(conf, metrics)| (*conf, metrics.recall))
            .collect();
        
        let f1_series: Vec<(f32, f32)> = conf_results.iter()
            .map(|(conf, metrics)| (*conf, metrics.f1_score))
            .collect();
        
        let iou_series: Vec<(f32, f32)> = conf_results.iter()
            .map(|(conf, metrics)| (*conf, metrics.iou))
            .collect();
        
        // Plot series
        chart.draw_series(LineSeries::new(precision_series, &RED))?
            .label("Precision")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        
        chart.draw_series(LineSeries::new(recall_series, &BLUE))?
            .label("Recall")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
        
        chart.draw_series(LineSeries::new(f1_series, &GREEN))?
            .label("F1 Score")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        
        chart.draw_series(LineSeries::new(iou_series, &MAGENTA))?
            .label("Avg IoU")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }
    
    // Create heat map for F1 scores across confidence_threshold and nms_threshold
    {
        // Find the best IOU threshold
        let best_iou = yolo_results.iter()
            .max_by(|a, b| a.1.f1_score.partial_cmp(&b.1.f1_score).unwrap())
            .map(|(p, _, _)| p.iou_threshold)
            .unwrap();
        
        let mut chart = ChartBuilder::on(lower)
            .caption(format!("YOLO F1 Score Heatmap (IoU Threshold = {:.1})", best_iou), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (conf_thresholds[0] as f64 - 0.05)..(conf_thresholds[conf_thresholds.len() - 1] as f64 + 0.05),
                (nms_thresholds[0] as f64 - 0.025)..(nms_thresholds[nms_thresholds.len() - 1] as f64 + 0.025)
            )?;
        
        chart.configure_mesh()
            .x_desc("Confidence Threshold")
            .y_desc("NMS Threshold")
            .disable_mesh()
            .draw()?;
        
        // Create heatmap data
        let cell_width = (conf_thresholds[1] as f64 - conf_thresholds[0] as f64 ) * 0.8;
        let cell_height = (nms_thresholds[1] as f64 - nms_thresholds[0] as f64 ) * 0.8;
        
        let mut heatmap_data = vec![];
        for &conf in &conf_thresholds {
            for &nms in &nms_thresholds {
                if let Some(entry) = yolo_results.iter().find(|(p, _, _)| 
                    (p.confidence_threshold - conf).abs() < 0.01 && 
                    (p.nms_threshold - nms).abs() < 0.01 && 
                    (p.iou_threshold - best_iou).abs() < 0.01
                ) {
                    heatmap_data.push((conf as f64, nms as f64, entry.1.f1_score as f64));
                }
            }
        }
        
        chart.draw_series(
            heatmap_data.iter().map(|&(x, y, f1)| {
                let color = HSLColor(
                    0.6 - f1 * 0.6, // Blue to red
                    0.8,
                    0.3 + f1 * 0.4,
                );
                Rectangle::new(
                    [(x - cell_width/2.0, y - cell_height/2.0), (x + cell_width/2.0, y + cell_height/2.0)],
                    color.filled(),
                )
            })
        )?;
        
        // Add F1 score labels
        chart.draw_series(
            heatmap_data.iter().map(|&(x, y, f1)| {
                Text::new(
                    format!("{:.2}", f1),
                    (x, y),
                    ("sans-serif", 15.0).into_font().color(&WHITE),
                )
            })
        )?;
    }
    
    Ok(())
}

// Save results to JSON file for later analysis
pub fn save_results(results: &HashMap<ModelType, (EvaluationMetrics, f32)>, filename: &str) -> Result<()> {
    #[derive(Serialize)]
    struct ParamResult {
        model_type: ModelType,
        metrics: (EvaluationMetrics, f32),
    }
    
    let serializable_results: Vec<ParamResult> = results.iter()
        .map(|(model_type, metrics)| ParamResult {
            model_type: *model_type,
            metrics: *metrics,
        })
        .collect();
    
    let json = serde_json::to_string_pretty(&serializable_results)?;
    std::fs::write(filename, json)?;
    
    Ok(())
}

// // Run grid search to find optimal parameters
// let results = grid_search("test_dataset")?;
    
// // Save raw results to JSON for potential further analysis
// save_results(&results, "param_search_results.json")?;

// // Generate visualization
// plot_evaluation_results(&results, "parameter_optimization.png")?;

// println!("Parameter search complete. Results saved to param_search_results.json");
// println!("Visualization saved to parameter_optimization.png");