// src/rtsp_processor.rs

use crate::detection::yolo::YoloDetector;
use crate::{detection, zone_manager};
use anyhow::{Result, bail};
use opencv::{
    core::{self, Mat},
    highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
};
use std::sync::mpsc::Sender;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};

// Struct to hold configurable parameters
#[derive(Clone)]
pub struct ProcessorConfig {
    pub rtsp_url: String,
    pub onnx_model_path: String,
    pub zones_path: String,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub show_gui: bool,
}

// The main struct that encapsulates the application logic
pub struct RTSPProcessor {
    config: Arc<Mutex<ProcessorConfig>>,
    exit_signal: Arc<AtomicBool>,
    processor_handle: Option<JoinHandle<()>>,
    pub camera_change_sender: Sender<u32>,
}

impl RTSPProcessor {
    /// Creates a new processor instance.
    pub fn new(config: ProcessorConfig, camera_change_sender: Sender<u32>) -> Self {
        Self {
            config: Arc::new(Mutex::new(config)),
            exit_signal: Arc::new(AtomicBool::new(false)),
            processor_handle: None,
            camera_change_sender, // Memorizza il mittente
        }
    }

    /// Starts the video processing in a new thread.
    pub fn run(&mut self) -> Result<()> {
        if self.processor_handle.is_some() {
            bail!("Processor is already running.");
        }

        let config = Arc::clone(&self.config);
        let exit_signal = Arc::clone(&self.exit_signal);
        // Sposta il mittente nel thread
        let sender = self.camera_change_sender.clone();

        let handle = thread::spawn(move || {
            if let Err(e) = run_processor_loop(config, exit_signal, sender) {
                eprintln!("Processor thread exited with error: {}", e);
            }
        });

        self.processor_handle = Some(handle);
        Ok(())
    }

    /// Signals the processor to stop and waits for it to finish.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(handle) = self.processor_handle.take() {
            println!("Sending stop signal to processor...");
            self.exit_signal.store(true, Ordering::SeqCst);
            handle.join().expect("Failed to join processor thread.");
            println!("Processor stopped.");
        }
        Ok(())
    }

    // --- Methods to modify config at runtime ---
    pub fn set_show_gui(&self, show: bool) {
        self.config.lock().unwrap().show_gui = show;
    }

    pub fn set_confidence_threshold(&self, threshold: f32) {
        println!("Setting confidence threshold to {}", threshold);
        self.config.lock().unwrap().confidence_threshold = threshold;
    }

    pub fn set_nms_threshold(&self, threshold: f32) {
        self.config.lock().unwrap().nms_threshold = threshold;
    }
}

/// The main processing logic, now encapsulated in a function.
fn run_processor_loop(
    config: Arc<Mutex<ProcessorConfig>>,
    exit_signal: Arc<AtomicBool>,
    cam_sender: Sender<u32>, // Nuovo parametro
) -> Result<()> {
    // Clone config at the beginning to avoid frequent locking
    let initial_config = config.lock().unwrap().clone();

    let yolo_detector = Arc::new(YoloDetector::new(
        &initial_config.onnx_model_path,
        initial_config.confidence_threshold,
        initial_config.nms_threshold,
    )?);
    let mut zone_manager_data = zone_manager::load_zones(&initial_config.zones_path)?;

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
        let current_config = config.lock().unwrap().clone(); // Get latest config

        if let Some(mut frame) = current_frame {
            let detections = yolo_detector.detect(&frame)?;
            let ball_detections: Vec<_> = detections
                .iter()
                .filter(|d| d.class_id == 0)
                .cloned()
                .collect();
            if !ball_detections.is_empty() {
                // Per semplicità, consideriamo solo il primo pallone rilevato
                let det = &ball_detections[0];
                let ball_position = opencv::core::Point::new(
                    ((det.bbox.x1 + det.bbox.x2) / 2.0) as i32,
                    det.bbox.y2 as i32
                );

                // --- LOGICA CHIAVE: USA LA TUA FUNZIONE E INVIA LA NOTIFICA ---
                let old_cam = zone_manager_data.old_zone().cam;
                let new_cam = zone_manager_data.get_cam(ball_position);

                // Se la telecamera è cambiata, invia un messaggio nel canale
                if new_cam != old_cam {
                    if let Err(e) = cam_sender.send(new_cam) {
                        eprintln!("Failed to send camera change notification: {}", e);
                    }
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

/// The reader thread function remains the same.
fn run_reader_thread(
    mut cap: VideoCapture,
    frame_mailbox: Arc<Mutex<Option<Mat>>>,
    exit_signal: Arc<AtomicBool>,
) {
    println!("[Reader Thread] Starting frame reading loop.");
    let mut frame = Mat::default();

    // Loop finché non riceviamo il segnale di uscita E possiamo leggere frame
    while !exit_signal.load(Ordering::SeqCst) && cap.read(&mut frame).unwrap_or(false) {
        if !frame.empty() {
            *frame_mailbox.lock().unwrap() = Some(frame.clone());
        }
    }
    println!("[Reader Thread] Exit signal received or stream ended. Releasing resources...");
    // `cap.release()` viene chiamato automaticamente quando `cap` esce dallo scope qui.
    // Aggiungiamo una chiamata esplicita per chiarezza e per essere sicuri.
    cap.release().expect("Failed to release reader capture");
    println!("[Reader Thread] Resources released.");
}
