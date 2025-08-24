use crate::detection::yolo::YoloDetector;
use crate::{detection};
use crate::rtsp_processor::zone_manager;
use anyhow::{Result, bail};
use opencv::{
    core::{self, Mat},
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};
use std::sync::mpsc::Sender;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use crate::rtsp_processor::zone_manager::{ZoneManager};

/// Holds configurable parameters for the processor.
#[derive(Clone)]
pub struct ProcessorConfig {
    pub rtsp_url: String,
    pub onnx_model_path: String,
    pub zones_path: String,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub show_gui: bool,
    pub change_threshold: f32,
    pub print_change_ball: bool
}

/// Encapsulates the main RTSP processing logic.
pub struct RTSPProcessor {
    config: Arc<Mutex<ProcessorConfig>>,
    exit_signal: Arc<AtomicBool>,
    processor_handle: Option<JoinHandle<()>>,
    zone_manager: Arc<Mutex<ZoneManager>>,
    pub camera_change_sender: Sender<u32>,
}

impl RTSPProcessor {
    /// Creates a new processor instance.
    pub fn new(config: ProcessorConfig, camera_change_sender: Sender<u32>) -> Self {
        Self {
            config: Arc::new(Mutex::new(config.clone())),
            exit_signal: Arc::new(AtomicBool::new(false)),
            zone_manager: Arc::new(Mutex::new(zone_manager::load_zones(&config.zones_path, config.change_threshold, config.print_change_ball).expect("error loading zones"))),
            processor_handle: None,
            camera_change_sender,
        }
    }

    /// Gets the ID of the currently active camera zone.
    pub fn get_active_cam(&self) -> u32 {
        let zm = self.zone_manager.lock().unwrap();
        zm.active_zone().cam
    }

    /// Starts the video processing in a new thread.
    pub fn run(&mut self) -> Result<()> {
        if self.processor_handle.is_some() {
            bail!("Processor is already running.");
        }

        let config = Arc::clone(&self.config);
        let exit_signal = Arc::clone(&self.exit_signal);
        let zone_manager = Arc::clone(&self.zone_manager);
        let sender = self.camera_change_sender.clone();

        let handle = thread::spawn(move || {
            if let Err(e) = run_processor_loop(zone_manager, config, exit_signal, sender) {
                eprintln!("Processor thread exited with error: {}", e);
            }
        });

        self.processor_handle = Some(handle);
        Ok(())
    }

    /// Signals the processor thread to stop and waits for it to finish.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(handle) = self.processor_handle.take() {
            println!("Sending stop signal to processor...");
            self.exit_signal.store(true, Ordering::SeqCst);
            handle.join().expect("Failed to join processor thread.");
            println!("Processor stopped.");
        }
        Ok(())
    }

    /// Sets whether the GUI window should be displayed.
    pub fn set_show_gui(&self, show: bool) {
        self.config.lock().unwrap().show_gui = show;
    }

    /// Sets the confidence threshold for object detection.
    pub fn set_confidence_threshold(&self, threshold: f32) {
        println!("Setting confidence threshold to {}", threshold);
        self.config.lock().unwrap().confidence_threshold = threshold;
    }

    /// Sets the Non-Maximum Suppression (NMS) threshold.
    pub fn set_nms_threshold(&self, threshold: f32) {
        self.config.lock().unwrap().nms_threshold = threshold;
    }
}

/// Runs the main loop for processing video frames, detecting objects, and managing zones.
fn run_processor_loop(
    zone_manager_data: Arc<Mutex<ZoneManager>>,
    config: Arc<Mutex<ProcessorConfig>>,
    exit_signal: Arc<AtomicBool>,
    cam_sender: Sender<u32>,
) -> Result<()> {
    let initial_config = config.lock().unwrap().clone();
    let mut zone_manager_data = zone_manager_data.lock().unwrap();
    let yolo_detector = Arc::new(YoloDetector::new(
        &initial_config.onnx_model_path,
        initial_config.confidence_threshold,
        initial_config.nms_threshold,
    )?);

    println!("Connecting to RTSP stream: {}", &initial_config.rtsp_url);
    let cap = videoio::VideoCapture::from_file(&initial_config.rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        bail!("Could not connect to the RTSP stream.");
    }
    println!("Successfully connected.");

    let latest_frame = Arc::new(Mutex::new(None::<Mat>));
    let reader_exit_signal = Arc::clone(&exit_signal);
    let mut show_zones_overlay = false;

    let reader_handle = thread::spawn({
        let frame_clone = Arc::clone(&latest_frame);
        move || {
            run_reader_thread(cap, frame_clone, reader_exit_signal);
        }
    });

    if initial_config.show_gui {
        highgui::named_window("RTSP Ball Detection", highgui::WINDOW_NORMAL)?;
    }

    while !exit_signal.load(Ordering::SeqCst) {
        let current_frame = { latest_frame.lock().unwrap().clone() };
        let current_config = config.lock().unwrap().clone();

        if let Some(mut frame) = current_frame {
            let detections = yolo_detector.detect(&frame)?;
            let ball_detections: Vec<_> = detections
                .iter()
                .filter(|d| d.class_id == 0)
                .cloned()
                .collect();
            let best_ball = ball_detections.iter().max_by(|a, b| a.confidence.unwrap_or(0.0).partial_cmp(&b.confidence.unwrap_or(0.0)).unwrap());
            let old_cam = zone_manager_data.active_zone().cam;
            if let Some(det) = best_ball {
                let ball_position = opencv::core::Point::new(
                    ((det.bbox.x1 + det.bbox.x2) / 2.0) as i32,
                    det.bbox.y2 as i32
                );
                zone_manager_data.update_active_cam(Some(ball_position))
            } else {
                zone_manager_data.update_active_cam(None)
            };
            let active_cam = zone_manager_data.active_zone().cam;
            if old_cam != active_cam {
                println!("Camera changed to: {}", active_cam);
                if let Err(e) = cam_sender.send(active_cam) {
                    eprintln!("Failed to send camera change notification: {}", e);
                }
            }

            if current_config.show_gui {
                if show_zones_overlay {
                    zone_manager_data.draw_zones_overlay(&mut frame)?;
                }
                detection::bounding_box::draw_boxes(
                    &mut frame,
                    &ball_detections,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                )?;
                highgui::imshow("RTSP Ball Detection", &frame)?;
                let key = highgui::wait_key(1)?;
                if key == 'q' as i32 {
                    exit_signal.store(true, Ordering::SeqCst);
                } else if key == 'o' as i32 {
                    show_zones_overlay = !show_zones_overlay;
                    println!(
                        "Zones overlay toggled to: {}",
                        if show_zones_overlay { "ON" } else { "OFF" }
                    );
                }
            }
        } else {
            thread::sleep(std::time::Duration::from_millis(10));
        }

        if reader_handle.is_finished() {
            println!("Reader thread has finished. Exiting main loop.");
            exit_signal.store(true, Ordering::SeqCst);
        }
    }

    highgui::destroy_all_windows()?;
    Ok(())
}

/// Reads frames from the video capture in a dedicated thread.
fn run_reader_thread(
    mut cap: VideoCapture,
    frame_mailbox: Arc<Mutex<Option<Mat>>>,
    exit_signal: Arc<AtomicBool>,
) {
    println!("[Reader Thread] Starting frame reading loop.");
    let mut frame = Mat::default();

    while !exit_signal.load(Ordering::SeqCst) && cap.read(&mut frame).unwrap_or(false) {
        if !frame.empty() {
            *frame_mailbox.lock().unwrap() = Some(frame.clone());
        }
    }
    println!("[Reader Thread] Exit signal received or stream ended. Releasing resources...");
    cap.release().expect("Failed to release reader capture");
    println!("[Reader Thread] Resources released.");
}