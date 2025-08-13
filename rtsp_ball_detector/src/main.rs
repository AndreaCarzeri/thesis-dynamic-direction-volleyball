// main.rs

mod detection;
mod zone_manager;
mod rtsp_processor;

use anyhow::{bail, Result};
use opencv::{core::{self, Mat}, highgui, imgproc, prelude::*, videoio::{self, VideoCapture}};
use detection::yolo::YoloDetector;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Duration;
use crate::rtsp_processor::{ProcessorConfig, RTSPProcessor};

fn main() -> Result<()> {
    // 1. Definisci la configurazione iniziale
    let config = ProcessorConfig {
        rtsp_url: "rtsp://127.0.0.1:8554/stream".to_string(),
        onnx_model_path: "assets/best.onnx".to_string(),
        zones_path: "assets/field_zones.json".to_string(),
        confidence_threshold: 0.6,
        nms_threshold: 0.5,
        show_gui: true,
    };
    let (tx, rx): (Sender<u32>, Receiver<u32>) = channel();

    // 2. Crea l'istanza del processore
    let mut processor = RTSPProcessor::new(config, tx);

    // 3. Avvia l'elaborazione (in background)
    processor.run()?;
    println!("Processor started. Running for 30 seconds before demonstrating config changes.");

    loop {
        // `try_recv()` è non-bloccante. Controlla se c'è un messaggio.
        match rx.try_recv() {
            Ok(new_cam_id) => {
                // Abbiamo ricevuto una notifica!
                println!("\n>>> MAIN: Received notification! Camera changed to: {} <<<\n", new_cam_id);
                // Qui puoi mettere qualsiasi logica vuoi eseguire quando la telecamera cambia.
            },
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Nessun nuovo messaggio, non fare nulla e continua a controllare
            },
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                println!("Processor thread has disconnected. Exiting listener.");
                break; // Esce dal loop se il processore si ferma
            }
        }

        // Aggiungiamo un piccolo sleep per evitare di consumare il 100% della CPU
        // in questo loop di ascolto.
        thread::sleep(Duration::from_millis(100));
    }

    // 4. Ferma l'elaborazione in modo pulito
    processor.stop()?;

    println!("Main application finished.");
    Ok(())
}