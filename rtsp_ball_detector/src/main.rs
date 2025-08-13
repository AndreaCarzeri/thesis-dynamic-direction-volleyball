// main.rs

mod detection;
mod zone_manager;

use anyhow::{bail, Result};
use opencv::{core::{self, Mat}, highgui, imgproc, prelude::*, videoio::{self, VideoCapture}};
use detection::yolo::YoloDetector;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> Result<()> {
    // --- 1. CONFIGURATION ---
    let rtsp_url = "rtsp://127.0.0.1:8554/stream";
    let onnx_model_path = "assets/best.onnx";
    let confidence_threshold = 0.6;
    let nms_threshold = 0.5;
    let zones_path = "assets/field_zones.json";
    let show_gui = true;

    // --- 2. INITIALIZE SHARED RESOURCES ---
    let yolo_detector = Arc::new(YoloDetector::new(onnx_model_path, confidence_threshold, nms_threshold)?);
    let mut zone_manager_data = zone_manager::load_zones(zones_path)?;
    println!("Model and zones loaded successfully.");

    // --- 3. CONNECT TO STREAM IN MAIN THREAD (CRUCIAL STEP) ---
    println!("Connecting to RTSP stream in main thread: {}", rtsp_url);
    let cap = videoio::VideoCapture::from_file(rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        bail!("Error: Could not connect to the RTSP stream in main thread.");
    }
    println!("Successfully connected.");

    // --- 4. SETUP SHARED STATE AND START READER THREAD ---
    let latest_frame = Arc::new(Mutex::new(None::<Mat>));
    let latest_frame_for_reader = Arc::clone(&latest_frame);

    // The opened 'cap' object is MOVED into the new thread.
    thread::spawn(move || {
        run_reader_thread(cap, latest_frame_for_reader);
    });

    println!("Reader thread started. Starting main processing loop.");
    if show_gui {
        highgui::named_window("RTSP Ball Detection", highgui::WINDOW_NORMAL)?;
        println!("GUI is enabled. Press 'q' in the window to exit.");
    }
    let mut frame_count = 0;
    // --- 5. MAIN PROCESSING LOOP (PROCESSOR THREAD) ---
    loop {
        // Lock the mutex briefly to get the latest frame
        let current_frame = {
            latest_frame.lock().unwrap().clone()
        };

        if let Some(mut frame) = current_frame {
            frame_count+=1;
            if frame_count % 5 != 0 {
                continue;
            }
            let detections = yolo_detector.detect(&frame)?;
            let ball_detections: Vec<_> = detections.iter().filter(|d| d.class_id == 0).cloned().collect();

            // Your terminal output logic
            if !ball_detections.is_empty() {
                println!("\nDetected {} ball(s):", ball_detections.len());
                for (i, det) in ball_detections.iter().enumerate() {
                    let ball_position_x = (det.bbox.x1 + det.bbox.x2) / 2.0;
                    let ball_position_y = det.bbox.y2;
                    let ball_position = opencv::core::Point::new(ball_position_x as i32, ball_position_y as i32);

                    let cam = zone_manager_data.get_cam(ball_position);
                    println!(
                        "  Ball {}: Position ({:.2}, {:.2}), Confidence: {:.2}, Zone ID: {}, Camera: {}",
                        i + 1,
                        ball_position_x,
                        ball_position_y,
                        det.confidence.unwrap_or(0.0),
                        zone_manager_data.old_zone().id,
                        cam
                    );
                }
            }

            if show_gui {
                // Your drawing logic
                detection::bounding_box::draw_boxes(&mut frame, &ball_detections, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2)?;
                for det in &ball_detections {
                    let ball_position_x = (det.bbox.x1 + det.bbox.x2) / 2.0;
                    let ball_position_y = det.bbox.y2;
                    let ball_position = opencv::core::Point::new(ball_position_x as i32, ball_position_y as i32);
                    imgproc::circle(&mut frame, ball_position, 5, opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0), -1, imgproc::LINE_AA, 0)?;
                }

                highgui::imshow("RTSP Ball Detection", &frame)?;
                if highgui::wait_key(1)? == 'q' as i32 {
                    break;
                }
            }
        } else {
            // If no frame is available yet, wait briefly
            thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    // --- 6. CLEANUP ---
    println!("Exiting application and releasing resources...");
    highgui::destroy_all_windows()?; // Explicitly destroy all GUI windows
    println!("Application finished.");

    Ok(())
}


/// The function that runs within the dedicated reader thread.
/// It receives an already opened VideoCapture object and continuously reads frames.
fn run_reader_thread(mut cap: VideoCapture, frame_mailbox: Arc<Mutex<Option<Mat>>>) {
    println!("[Reader Thread] Starting frame reading loop.");
    let mut frame = Mat::default();

    // Loop as long as we can read frames. The `?` operator is not used here
    // to prevent the thread from panicking on a read error.
    while cap.read(&mut frame).unwrap_or(false) {
        if !frame.empty() {
            // Lock, put the new frame in the mailbox, and release.
            *frame_mailbox.lock().unwrap() = Some(frame.clone());
        }
    }

    println!("[Reader Thread] Stream ended or connection lost. Releasing resources.");
    let _ = cap.release();
    // The `cap` object goes out of scope here, and Rust automatically calls its
    // destructor, which in turn calls `release()` on the underlying OpenCV object.
    // An explicit call to `cap.release()` is not strictly necessary but can be added for clarity.
    // cap.release().expect("Failed to release reader capture");
}