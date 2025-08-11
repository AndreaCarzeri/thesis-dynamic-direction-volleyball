use std::{collections::HashMap, sync::Arc};
use opencv::{core::{Mat, Point, Scalar}, highgui, imgcodecs, imgproc, prelude::*, videoio::{self, VideoCapture}};
use ort::{Environment, Session, SessionBuilder, Value};
use ndarray::{Array, Array4, IxDyn, s, Axis, CowArray};
use anyhow::{Context, Result};
use opencv::core::AlgorithmHint;
use super::{Detection, bounding_box::{self, BoundingBox}};

pub struct YoloDetector {
    session: ort::Session,
    input_shape: (usize, usize),
	precision_threshold: f32,
	nms_threshold: f32,
}

impl YoloDetector {
    pub fn new(model_path: &str, precision_threshold: f32, nms_threshold: f32) -> Result<Self> {
        let env = Arc::new(Environment::builder()
			.with_name("yolov8n")
            .build()
			.context("Failed to load the model")?);

        let session = SessionBuilder::new(&env)
			.context("Failed to build session")?
            .with_model_from_file(model_path)
			.context("Failed to load model file")?;

        Ok(Self {
            session,
            input_shape: (640, 640),
			precision_threshold,
			nms_threshold,
        })
    }

	pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
		let (preprocessed, nwe_height, new_width) = self.preprocess(image).context("Error during preprocessing")?;
        // Convert Array4 to dynamic-dimensional array
    	let input_array = preprocessed.into_dyn();
		let cow_array = CowArray::from(input_array);

        let input_tensor = Value::from_array(self.session.allocator(), &cow_array).context("Error during to input tensor")?;

        let outputs = self.session.run(vec![input_tensor]).context("Error during processing of the input")?;

		let output = outputs.get(0).unwrap().try_extract::<f32>().unwrap().view().t().into_owned();

        self.process_output(&output, (image.cols(), image.rows()), (nwe_height, new_width)).context("Error during processing of the output")
    }

	fn preprocess(&self, image: &Mat) -> Result<(Array4<f32>, i32, i32)> {
        let mut rgb = Mat::default();
		opencv::imgproc::cvt_color(image, &mut rgb, opencv::imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT).context("Error during conversion to RGB")?;

		// Maintain aspect ratio while resizing to closest multiple of 32
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
        ).context("Error during resizing of the image")?;

        let mut letterboxed = Mat::new_rows_cols_with_default(
            self.input_shape.0 as i32,
            self.input_shape.1 as i32,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::new(114.0, 114.0, 114.0, 0.0)
        ).context("Error during letterboxing of the image")?;

        let dw = (self.input_shape.1 as i32 - new_width as i32) / 2;
        let dh = (self.input_shape.0 as i32 - new_height as i32) / 2;

		let err_msg = format!("Error when taking roi of letterboxed image: {:?}, {}, {}, {}, {}", letterboxed.clone(), dw, dh, new_width, new_height);

        let mut roi = Mat::roi_mut(&mut letterboxed, opencv::core::Rect::new(
            dw, dh, new_width as i32, new_height as i32
        )).context(err_msg)?;
        resized.copy_to(&mut roi).context("Error when taking the roi of the resized function")?;

        let mut float_img = Mat::default();
        letterboxed.convert_to(&mut float_img, opencv::core::CV_32FC3, 1.0/255.0, 0.0).context("Error during conversion to float matrix")?;

		// let pixels = float_img.data_typed::<f32>().context("Error during conversion to array")?; -> problem with channels
		// let array = Array4::from_shape_vec(
		// 	(1, 3, self.input_shape.0, self.input_shape.1),
		// 	pixels.to_vec()
		// ).context("Error during reshaping to 4D array")?;

		// Split channels
		let mut channels = opencv::core::Vector::<Mat>::new();
		opencv::core::split(&float_img, &mut channels)?;

		// Create a vector to store the data from all channels
		let mut all_data = Vec::new();

		// Read data from each channel
		for channel in channels.iter() {
			let channel_data = channel.data_typed::<f32>()?;
			all_data.extend_from_slice(channel_data);
		}

		// Create the final array in NCHW format
		let array = Array4::from_shape_vec(
			(1, 3, self.input_shape.0, self.input_shape.1),
			all_data
		).context("Error during reshaping to 4D array")?;

        // Reshape to get contiguous channels
        // let pixel_count = (self.input_shape.0 * self.input_shape.1) as i32;
        // let reshaped = float_img.reshape(1, pixel_count * 3)?;
        // let pixels = reshaped.data_typed::<f32>()?;

        // // Create array in CHW format (needed for ONNX)
        // let array: Array4<f32> = Array::from_shape_vec(
        //     (1, 3, self.input_shape.0, self.input_shape.1),
        //     pixels.to_vec()
        // )?;

        Ok((array, new_height as i32, new_width as i32))
    }

	fn process_output(&self, output: &Array<f32,IxDyn>, img_size: (i32, i32), new_img_size: (i32, i32)) -> Result<Vec<Detection>> {
		let mut detections = Vec::new();
		let output = output.slice(s![..,..,0]);

		for row in output.axis_iter(Axis(0)) {
			let row:Vec<_> = row.iter().map(|x| *x).collect();
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

			// boxes.push((x1,y1,x2,y2,label,prob));
			detections.push(Detection::new(BoundingBox::new(x1, y1, x2, y2), Some(prob), class_id as i32));
		}

		let final_detections = bounding_box::non_maximum_suppression(&mut detections, self.nms_threshold);
		Ok(final_detections)
	}

	pub fn read_video(&self, video_path: &str) -> Result<()> {
		let mut cap = VideoCapture::from_file(video_path, videoio::CAP_ANY).context("Unable to open video!")?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = Mat::default();

		highgui::named_window("Frame", highgui::WINDOW_NORMAL)?;
		// highgui::named_window("FG Mask", highgui::WINDOW_NORMAL)?;

		let mut frame_count = 0;
		loop {
			cap.read(&mut frame)?;
			if frame.size()?.width == 0 {
				println!("End of video stream");
				break; // End of video stream.
			}
			frame_count += 1;

			imgproc::put_text(
				&mut frame,
				&format!("Frame: {}", frame_count),
				Point::new(10, 30),
				imgproc::FONT_HERSHEY_SIMPLEX,
				1.0,
				Scalar::new(0.0, 255.0, 0.0, 0.0),
				2,
				imgproc::LINE_AA,
				false
			)?;

			let detections = self.detect(&frame)?;

			bounding_box::draw_boxes(&mut frame, &detections, Scalar::new(0.0, 255.0, 0.0, 0.0), 2)?;
			highgui::imshow("Frame", &frame)?;

			// Break on key press (e.g., 'q' or any key with a delay).
			if highgui::wait_key(30)? > 0 {
				break;
			}
		}
		Ok(())
	}
	pub fn detect_video(&self, video_path: &str, frame_count: usize) -> Result<HashMap<usize, Vec<Detection>>> {
		let mut cap = VideoCapture::from_file(video_path, videoio::CAP_ANY).context("Unable to open video!")?;
		if !cap.is_opened()? {
			anyhow::bail!("Unable to open video!");
		}

		let mut frame = Mat::default();
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