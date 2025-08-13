// main.rs

mod detection;
mod zone_manager;

use anyhow::{bail, Result};
use opencv::{core::{self, Mat}, highgui, imgproc, prelude::*, videoio};
use detection::yolo::YoloDetector;
use std::sync::Arc;

fn main() -> Result<()> {
    // --- 1. CONFIGURATION ---
    let rtsp_url = "rtsp://127.0.0.1:8554/mystream";
    let onnx_model_path = "assets/best.onnx";
    let confidence_threshold = 0.4;
    let nms_threshold = 0.5;
    let zones_path = "assets/field_zones.json";

    // Set to true to display a window with detections, false to only print to console
    let show_gui = true;
    const RECONNECT_DELAY_SECS: u64 = 5;

    // --- 2. INITIALIZE ZONES AND YOLO DETECTOR ---
    // Load zones once at the beginning
    let zones = zone_manager::load_zones(zones_path)?;

    println!("Loading YOLO model from: {}", onnx_model_path);
    let yolo_detector = YoloDetector::new(onnx_model_path, confidence_threshold, nms_threshold)?;
    println!("Model loaded successfully.");

    // --- 3. MAIN APPLICATION LOOP WITH RECONNECTION LOGIC ---
    loop { // This outer loop handles reconnections
        println!("Attempting to connect to RTSP stream: {}", rtsp_url);

        match videoio::VideoCapture::from_file(rtsp_url, videoio::CAP_FFMPEG) {
            Ok(mut cap) => {
                if cap.is_opened()? {
                    println!("Successfully connected. Starting detection loop...");
                    if show_gui {
                        highgui::named_window("RTSP Ball Detection", highgui::WINDOW_NORMAL)?;
                    }

                    // Once connected, enter the inner processing loop
                    if let Err(e) = process_stream(&mut cap, &yolo_detector, &zones, show_gui) {
                        eprintln!("Error during stream processing: {}", e);
                    }

                    println!("Stream ended or was disconnected.");
                } else {
                    eprintln!("Could not open stream, although capture object was created.");
                }
            },
            Err(e) => {
                eprintln!("Failed to create VideoCapture object: {}", e);
            }
        }

        // If the stream ends or connection fails, wait before trying again.
        println!("Will attempt to reconnect in {} seconds...", RECONNECT_DELAY_SECS);
        std::thread::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS));
    }
}

/// Processes frames from an active and opened video stream.
fn process_stream(
    cap: &mut videoio::VideoCapture,
    yolo_detector: &YoloDetector,
    zones: &[zone_manager::Zone],
    show_gui: bool,
) -> Result<()> {
    let mut frame = Mat::default();
    let mut frame_count = 0;

    while cap.read(&mut frame)? {
        if frame.empty() {
            println!("Received an empty frame, continuing...");
            continue;
        }
        frame_count += 1;

        // Perform detection using your YoloDetector
        let detections = yolo_detector.detect(&frame)?;

        // Filter for the "volleyball" class (ID 0)
        let ball_detections: Vec<_> = detections.iter()
            .filter(|d| d.class_id == 0)
            .cloned()
            .collect();

        // Print results to the terminal
        println!("--- Frame {} ---", frame_count);
        if ball_detections.is_empty() {
            println!("No balls detected.");
        } else {
            println!("Detected {} ball(s):", ball_detections.len());
            for (i, det) in ball_detections.iter().enumerate() {
                let ball_position_x = (det.bbox.x1 + det.bbox.x2) / 2.0;
                let ball_position_y = det.bbox.y2;
                let ball_position = opencv::core::Point::new(ball_position_x as i32, ball_position_y as i32);

                // Check which zone the ball is in
                if let Some(zone_id) = zone_manager::get_zone_for_point(ball_position, zones) {
                    println!(
                        "  Ball #{}: Zone ID: {} at ground position ({}, {})",
                        i + 1, zone_id, ball_position.x, ball_position.y
                    );
                } else {
                    println!(
                        "  Ball #{}: Detected at ({}, {}), but it's outside any defined zone.",
                        i + 1, ball_position.x, ball_position.y
                    );
                }
            }
        }

        // Display the results in a GUI window if enabled
        if show_gui {
            let mut display_frame = frame.clone();
            detection::bounding_box::draw_boxes(&mut display_frame, &ball_detections, opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2)?;

            // Draw the ground position point for debugging
            for det in &ball_detections {
                let ball_position_x = (det.bbox.x1 + det.bbox.x2) / 2.0;
                let ball_position_y = det.bbox.y2;
                let ball_position = opencv::core::Point::new(ball_position_x as i32, ball_position_y as i32);
                imgproc::circle(&mut display_frame, ball_position, 5, opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0), -1, imgproc::LINE_AA, 0)?;
            }

            highgui::imshow("RTSP Ball Detection", &display_frame)?;

            // Exit cleanly if 'q' is pressed
            if highgui::wait_key(1)? == 'q' as i32 {
                println!("'q' pressed, exiting.");
                return Ok(());
            }
        }
    }

    // This is reached when cap.read() returns false, indicating end of stream
    Ok(())
}