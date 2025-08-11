// main.rs

mod detection;

use anyhow::Result;
use opencv::{core::Mat, highgui, prelude::*, videoio};
use detection::yolo::YoloDetector;

fn main() -> Result<()> {
    // --- 1. CONFIGURATION ---
    let rtsp_url = "rtsp://127.0.0.1:8554/mystream";
    let onnx_model_path = "assets/best.onnx";
    let confidence_threshold = 0.4;
    let nms_threshold = 0.5;

    // Set to true to display a window with detections, false to only print to console
    let show_gui = false;

    // --- 2. INITIALIZE YOLO DETECTOR ---
    // Uses the YoloDetector struct from your `detection/yolo.rs` file
    println!("Loading YOLO model from: {}", onnx_model_path);
    let yolo_detector = YoloDetector::new(onnx_model_path, confidence_threshold, nms_threshold)?;
    println!("Model loaded successfully.");

    // --- 3. CONNECT TO RTSP STREAM ---
    println!("Connecting to RTSP stream: {}", rtsp_url);
    let mut cap = videoio::VideoCapture::from_file(rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        anyhow::bail!("Error: Could not connect to the RTSP stream.");
    }
    println!("Successfully connected. Starting detection loop...");
    if show_gui {
        highgui::named_window("RTSP Ball Detection", highgui::WINDOW_NORMAL)?;
    }

    // --- 4. REAL-TIME PROCESSING LOOP ---
    let mut frame = Mat::default();
    let mut frame_count = 0;

    loop {
        if !cap.read(&mut frame)? || frame.empty() {
            println!("Warning: Could not read frame. Stream may have ended or been interrupted.");
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }
        frame_count += 1;

        // Use your existing `detect` method
        let detections = yolo_detector.detect(&frame)?;

        // --- 5. PRINT RESULTS TO TERMINAL ---
        println!("--- Frame {} ---", frame_count);

        let ball_detections: Vec<_> = detections.iter()
            .filter(|d| d.class_id == 0).cloned() // Filter for class_id 0 ("volleyball")
            .collect();

        if ball_detections.is_empty() {
            println!("No balls detected.");
        } else {
            println!("Detected {} ball(s):", ball_detections.len());
            for (i, det) in ball_detections.iter().enumerate() {
                println!(
                    "  Ball #{}: Coordinates [x1: {:.1}, y1: {:.1}, x2: {:.1}, y2: {:.1}], Confidence: {:.2}",
                    i + 1,
                    det.bbox.x1,
                    det.bbox.y1,
                    det.bbox.x2,
                    det.bbox.y2,
                    det.confidence.unwrap_or(0.0)
                );
            }
        }

        // --- 6. (OPTIONAL) DISPLAY GUI ---
        if show_gui {
            // Draw detections on the frame for visual feedback
            detection::bounding_box::draw_boxes(&mut frame, &ball_detections, opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2)?;
            highgui::imshow("RTSP Ball Detection", &frame)?;

            // Exit if 'q' is pressed in the GUI window
            if highgui::wait_key(1)? == 'q' as i32 {
                break;
            }
        }
    }

    Ok(())
}