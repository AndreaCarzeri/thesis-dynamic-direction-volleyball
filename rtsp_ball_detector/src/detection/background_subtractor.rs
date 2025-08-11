use std::collections::HashMap;

use opencv::{
    core::{self, Point, Ptr, Scalar}, highgui, imgcodecs, imgproc, prelude::*, video::{
		self, BackgroundSubtractorMOG2Trait
	}, videoio::{self, VideoCapture}
};
use anyhow::{Result, Context};
use opencv::core::AlgorithmHint;
use super::{Detection, bounding_box};

fn preprocess(mask: &Mat) -> Result<Mat> {       
	highgui::named_window("Mask", highgui::WINDOW_NORMAL)?;
    highgui::imshow("Mask", &mask)?;
	
	// Apply median blur with a kernel size of 11
    let mut blurred = Mat::default();
    imgproc::median_blur(mask, &mut blurred, 11)?;

	highgui::named_window("blurred", highgui::WINDOW_NORMAL)?;
	highgui::imshow("blurred", &blurred)?;

    // Create a 3x3 rectangular structuring element
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(3, 3),
        Point::new(-1, -1),
    )?;

    // Apply erosion with 2 iterations
    let mut eroded = Mat::default();
    imgproc::erode(
        &blurred,
        &mut eroded,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

	highgui::named_window("eroded", highgui::WINDOW_NORMAL)?;
	highgui::imshow("eroded", &eroded)?;

    let mut dilated = Mat::default();
    imgproc::dilate(
        &eroded,
        &mut dilated,
        &kernel,
        Point::new(-1, -1),
        7, // 18
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

	highgui::named_window("dilated", highgui::WINDOW_NORMAL)?;
	highgui::imshow("dilated", &dilated)?;

    // Apply a second erosion with 2 iterations
    let mut final_eroded = Mat::default();
    imgproc::erode(
        &dilated,
        &mut final_eroded,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

	highgui::named_window("final_eroded", highgui::WINDOW_NORMAL)?;
	highgui::imshow("final_eroded", &final_eroded)?;
	// if highgui::wait_key(0)? == 's' as i32 {
	// 	let steps = vec![
	// 		("mask", mask),
	// 		("blurred", &blurred),
	// 		("eroded", &eroded),
	// 		("dilated", &dilated),
	// 		("final_eroded", &final_eroded)
	// 	];

	// 	for (name, img) in steps {
	// 		let filename = format!("{}_{}.png", name, 0);
	// 		imgcodecs::imwrite(&filename, img, &core::Vector::<i32>::new())?;
	// 		println!("Saved {} to {}", name, filename);
	// 	}
	// }
	// highgui::wait_key(0)?;

	// highgui::destroy_window("Mask")?;
	// highgui::destroy_window("blurred")?;
	// highgui::destroy_window("eroded")?;
	// highgui::destroy_window("dilated")?;
	// highgui::destroy_window("final_eroded")?;

    Ok(final_eroded)
}

pub trait BackgroundSubtractor {
	fn new() -> Result<Self> where Self: Sized;
	fn detect(&mut self, frame: &core::Mat) -> Result<Vec<Detection>>;
	fn detect_video(&mut self, video_path: &str, frame_count: usize) -> Result<HashMap<usize, Vec<Detection>>> {
		let mut cap = VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = core::Mat::default();
		let mut frame_number = 0;
		let mut detections: HashMap<usize, Vec<Detection>> = HashMap::new();
		loop {
			cap.read(&mut frame)?;
			if frame.size()?.width == 0 {
				break; // End of video stream.
			}
			frame_number += 1;
			let frame_detections = self.detect(&frame)?;
			
			detections.entry(frame_number).or_insert(Vec::new()).extend_from_slice(&frame_detections);

			if frame_number >= frame_count {
				break;	
			}
		}

		Ok(detections)
	}
}

pub struct MOG2 {
	mog2: Ptr<video::BackgroundSubtractorMOG2>
}

impl MOG2 {
	pub fn new_with_par(var_threshold: f64, nMixtures: i32) -> Result<Self> {
		let mut mog2 = video::create_background_subtractor_mog2(500, var_threshold, false)?;
		mog2.set_n_mixtures(nMixtures)?;
		let mut cap = VideoCapture::from_file("../assets/background.mp4", videoio::CAP_ANY)?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = core::Mat::default();
		let mut fg_mask = core::Mat::default();
		loop {
			cap.read(&mut frame)?;
			if frame.size()?.width == 0 {
				break; // End of video stream.
			}
			// Apply the subtractor to update its model and produce a foreground mask.
			BackgroundSubtractorMOG2Trait::apply(&mut mog2, &frame, &mut fg_mask, 0.01)?;
		}

		Ok(MOG2 { mog2 })
	}
}

impl BackgroundSubtractor for MOG2 {
	fn new() -> Result<Self> {
		let mut mog2 = video::create_background_subtractor_mog2(500, 150.0, false)?;
		mog2.set_n_mixtures(3)?;
		let mut cap = VideoCapture::from_file("../assets/background.mp4", videoio::CAP_ANY)?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = core::Mat::default();
		let mut fg_mask = core::Mat::default();
		loop {
			cap.read(&mut frame)?;
			if frame.size()?.width == 0 {
				break; // End of video stream.
			}
			// Apply the subtractor to update its model and produce a foreground mask.
			BackgroundSubtractorMOG2Trait::apply(&mut mog2, &frame, &mut fg_mask, 0.01)?;
		}

		Ok(MOG2 { mog2 })
	} 
	fn detect(&mut self, frame: &core::Mat) -> Result<Vec<Detection>> {
		let mut fg_mask = core::Mat::default();
		BackgroundSubtractorMOG2Trait::apply(&mut self.mog2, &frame, &mut fg_mask, 0.0)?;
		fg_mask = preprocess(&fg_mask)?;
		let detections: Vec<Detection> = bounding_box::extract_boxes(&fg_mask)?;

		Ok(detections)
	}
}

pub struct KNN {
	knn: Ptr<video::BackgroundSubtractorKNN>
}

impl KNN {
	pub fn new_with_par(dist2_threshold: f64, knn_samples: i32) -> Result<Self> {
		let mut knn = video::create_background_subtractor_knn(500, dist2_threshold, false)?;
		knn.setk_nn_samples(knn_samples)?;
		let mut cap = VideoCapture::from_file("../assets/background.mp4", videoio::CAP_ANY)?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = core::Mat::default();
		let mut fg_mask = core::Mat::default();
		loop {
			cap.read(&mut frame)?;
			if frame.size()?.width == 0 {
				break; // End of video stream.
			}
			// Apply the subtractor to update its model and produce a foreground mask.
			knn.apply(&frame, &mut fg_mask, 0.01)?;
		}

		Ok(KNN { knn })
	}
}

impl BackgroundSubtractor for KNN {
	fn new() -> Result<Self> {
		let mut knn = video::create_background_subtractor_knn(500, 1200.0, false)?;
		knn.setk_nn_samples(4)?;
		let mut cap = VideoCapture::from_file("../assets/background.mp4", videoio::CAP_ANY)?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = core::Mat::default();
		let mut fg_mask = core::Mat::default();
		loop {
			cap.read(&mut frame)?;
			if frame.size()?.width == 0 {
				break; // End of video stream.
			}
			// Apply the subtractor to update its model and produce a foreground mask.
			knn.apply(&frame, &mut fg_mask, 0.01)?;
		}

		Ok(KNN { knn })
	} 
	fn detect(&mut self, frame: &core::Mat) -> Result<Vec<Detection>> {
		let mut fg_mask = core::Mat::default();
		self.knn.apply(&frame, &mut fg_mask, 0.0)?;
		fg_mask = preprocess(&fg_mask)?;
		let detections: Vec<Detection> = bounding_box::extract_boxes(&fg_mask)?;

		Ok(detections)
	}
}

/// Performs static background subtraction.
/// 
/// # Arguments
/// - `frame`: The current frame (Mat) to process.
/// - `background_path`: File path to the reference image (empty field).
/// - `threshold_value`: The threshold value to binarize the difference image (e.g., 30.0).
///
/// # Returns
/// A binary mask (Mat) where the foreground (moving objects) is white (255) and the background is black.
pub fn static_background_subtraction(frame: &core::Mat, background_path: &str, threshold_value: f64) -> Result<core::Mat> {
    // Load the background (empty field) image.
    let background = imgcodecs::imread(background_path, imgcodecs::IMREAD_COLOR)?;
    
    // Ensure the background image has the same size as the frame.
    let mut resized_background = core::Mat::default();
    if background.size()? != frame.size()? {
        imgproc::resize(&background, &mut resized_background, frame.size()?, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    } else {
        resized_background = background;
    }
    
    // Convert both the frame and the background to grayscale.
    let mut gray_frame = core::Mat::default();
    let mut gray_background = core::Mat::default();
    imgproc::cvt_color(frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
    imgproc::cvt_color(&resized_background, &mut gray_background, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // Compute the absolute difference between the background and the current frame.
    let mut diff = core::Mat::default();
    core::absdiff(&gray_background, &gray_frame, &mut diff)?;
    
    // Threshold the difference image to obtain a binary mask.
    let mut mask = core::Mat::default();
    imgproc::threshold(&diff, &mut mask, threshold_value, 255.0, imgproc::THRESH_BINARY)?;
    
    // Apply morphological opening (erosion followed by dilation) to reduce noise.
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(5, 5),
        core::Point::new(-1, -1)
    )?;
    let mut clean_mask = core::Mat::default();
    imgproc::morphology_ex(&mask, &mut clean_mask, imgproc::MORPH_OPEN, &kernel, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, core::Scalar::default())?;
    
    Ok(clean_mask)
}