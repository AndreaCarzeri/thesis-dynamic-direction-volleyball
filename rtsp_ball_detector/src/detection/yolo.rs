use std::sync::Arc;
use opencv::{core::{Mat},prelude::*};
use ort::{Environment, SessionBuilder, Value};
use ndarray::{Array, Array4, Axis, CowArray, IxDyn, s};
use anyhow::{Result};
use opencv::core::AlgorithmHint;
use super::{Detection, bounding_box::{self, BoundingBox}};

/// A detector for YOLO models using the ONNX Runtime.
pub struct YoloDetector {
	session: ort::Session,
	input_shape: (usize, usize),
	precision_threshold: f32,
	nms_threshold: f32,
}

impl YoloDetector {
	/// Creates a new YoloDetector instance from a model path.
	pub fn new(model_path: &str, precision_threshold: f32, nms_threshold: f32) -> Result<Self> {
		let env = Arc::new(Environment::builder()
			.with_name("yolo_detector")
			.build()?);

		let session = SessionBuilder::new(&env)?
			.with_model_from_file(model_path)?;

		Ok(Self {
			session,
			input_shape: (640, 640),
			precision_threshold,
			nms_threshold,
		})
	}

	/// Performs object detection on a given image.
	pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
		let (preprocessed, new_height, new_width) = self.preprocess(image)?;
		let input_array = preprocessed.into_dyn();
		let cow_array = CowArray::from(input_array);
		let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;
		let outputs = self.session.run(vec![input_tensor])?;
		let output = outputs.first().unwrap().try_extract::<f32>().unwrap().view().t().into_owned();
		self.process_output(&output, (image.cols(), image.rows()), (new_height, new_width))
	}

	/// Preprocesses an image for model input, including resizing and letterboxing.
	fn preprocess(&self, image: &Mat) -> Result<(Array4<f32>, i32, i32)> {
		let mut rgb = Mat::default();
		opencv::imgproc::cvt_color(image, &mut rgb, opencv::imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

		let scale = (self.input_shape.0 as f32 / image.rows() as f32).min(self.input_shape.1 as f32 / image.cols() as f32);
		let new_height = (image.rows() as f32 * scale / 32.0).round() * 32.0;
		let new_width = (image.cols() as f32 * scale / 32.0).round() * 32.0;

		let mut resized = Mat::default();
		opencv::imgproc::resize(
			&rgb,
			&mut resized,
			opencv::core::Size::new(new_width as i32, new_height as i32),
			0.0,
			0.0,
			opencv::imgproc::INTER_LINEAR
		)?;

		let mut letterboxed = Mat::new_rows_cols_with_default(
			self.input_shape.0 as i32,
			self.input_shape.1 as i32,
			opencv::core::CV_8UC3,
			opencv::core::Scalar::new(114.0, 114.0, 114.0, 0.0)
		)?;

		let dw = (self.input_shape.1 as i32 - new_width as i32) / 2;
		let dh = (self.input_shape.0 as i32 - new_height as i32) / 2;
		let mut roi = Mat::roi_mut(&mut letterboxed, opencv::core::Rect::new(dw, dh, new_width as i32, new_height as i32))?;
		resized.copy_to(&mut roi)?;

		let mut float_img = Mat::default();
		letterboxed.convert_to(&mut float_img, opencv::core::CV_32FC3, 1.0/255.0, 0.0)?;

		let mut channels = opencv::core::Vector::<Mat>::new();
		opencv::core::split(&float_img, &mut channels)?;

		let mut all_data = Vec::new();
		for channel in channels.iter() {
			let channel_data = channel.data_typed::<f32>()?;
			all_data.extend_from_slice(channel_data);
		}

		let array = Array4::from_shape_vec(
			(1, 3, self.input_shape.0, self.input_shape.1),
			all_data
		)?;

		Ok((array, new_height as i32, new_width as i32))
	}

	/// Processes the model's raw output into a list of detections.
	fn process_output(&self, output: &Array<f32,IxDyn>, img_size: (i32, i32), new_img_size: (i32, i32)) -> Result<Vec<Detection>> {
		let mut detections = Vec::new();
		let output = output.slice(s![..,..,0]);

		for row in output.axis_iter(Axis(0)) {
			let row:Vec<_> = row.iter().copied().collect();
			let (class_id, prob) = row.iter().skip(4).enumerate()
				.map(|(index, value)| (index, *value))
				.reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
				.unwrap();

			if prob < self.precision_threshold {
				continue
			}

			let scale = (self.input_shape.0 as f32 / img_size.0 as f32).min(self.input_shape.1 as f32 / img_size.1 as f32);
			let dw = (self.input_shape.1 as i32 - new_img_size.1) / 2;
			let dh = (self.input_shape.0 as i32 - new_img_size.0) / 2;

			let xc = (row[0] - dw as f32) / scale;
			let yc = (row[1] - dh as f32) / scale;
			let w = row[2] / scale;
			let h = row[3] / scale;

			let x1 = xc - w / 2.0;
			let x2 = xc + w / 2.0;
			let y1 = yc - h / 2.0;
			let y2 = yc + h / 2.0;

			detections.push(Detection::new(BoundingBox::new(x1, y1, x2, y2), Some(prob), class_id as i32));
		}

		let final_detections = bounding_box::non_maximum_suppression(&mut detections, self.nms_threshold);
		Ok(final_detections)
	}
	
}