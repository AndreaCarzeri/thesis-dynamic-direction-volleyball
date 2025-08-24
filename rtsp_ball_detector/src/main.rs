mod detection;
mod rtsp_processor;

use anyhow::Result;
use std::sync::mpsc::{channel, Receiver, Sender};
use crate::rtsp_processor::processor::{ProcessorConfig, RTSPProcessor};

/// The main entry point of the application.
fn main() -> Result<()> {
    // Define the initial configuration for the RTSP processor.
    let config = ProcessorConfig {
        rtsp_url: "rtsp://127.0.0.1:8554/stream".to_string(),
        onnx_model_path: "assets/best-small.onnx".to_string(),
        zones_path: "assets/field_zones.json".to_string(),
        confidence_threshold: 0.4,
        nms_threshold: 0.5,
        show_gui: true,
        change_threshold: 1.0, // in seconds
        print_change_ball: true
    };

    // Create a channel to receive notifications about camera changes from the processor.
    let (tx, rx): (Sender<u32>, Receiver<u32>) = channel();

    // Create an instance of the processor.
    let mut processor = RTSPProcessor::new(config, tx);

    // Start the processing in a background thread.
    processor.run()?;

    // Loop to listen for messages from the processor thread.
    loop {
        // Perform a non-blocking check for a new message.
        match rx.try_recv() {
            Ok(new_cam_id) => {
                // A notification was received.
                println!("\n>>> MAIN: Received notification! Camera changed to: {} <<<\n", new_cam_id);
            },
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // No new message, continue the loop.
            },
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                println!("Processor thread has disconnected. Exiting listener.");
                break; // Exit the loop if the processor stops.
            }
        }
    }

    // Stop the processing cleanly.
    processor.stop()?;

    println!("Main application finished.");
    Ok(())
}