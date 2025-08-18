use std::fs;
use std::io::{self, Write};
use std::path::Path;

use opencv::{
    core,
    imgcodecs,
    prelude::*,
    videoio::{self, VideoCapture},
};

// --- SETTINGS ---
const VIDEO_PATH: &str = "assets/out6.mp4";
const OUTPUT_DIR: &str = "extracted_frames";
const NUM_FRAMES_TO_EXTRACT: i64 = 500;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Rust Frame Extractor ---");

    // Check if the video file exists
    if !Path::new(VIDEO_PATH).exists() {
        return Err(format!("Error: Video file not found at path: {}", VIDEO_PATH).into());
    }

    // Clean and create the output directory
    if Path::new(OUTPUT_DIR).exists() {
        println!("Folder '{}' already exists. Cleaning it up...", OUTPUT_DIR);
        fs::remove_dir_all(OUTPUT_DIR)?;
    }
    fs::create_dir_all(OUTPUT_DIR)?;
    println!("Folder '{}' created.", OUTPUT_DIR);

    // Open the video
    let mut cap = VideoCapture::from_file(VIDEO_PATH, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err("Error: Could not open the video file.".into());
    }

    // Get video properties
    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i64;

    // Calculate the frame extraction interval
    let interval = if total_frames < NUM_FRAMES_TO_EXTRACT {
        println!(
            "Warning: Video has only {} frames, less than the requested {}. Extracting all available frames.",
            total_frames, NUM_FRAMES_TO_EXTRACT
        );
        1
    } else {
        total_frames / NUM_FRAMES_TO_EXTRACT
    };

    println!("Video loaded: {} total frames.", total_frames);
    println!(
        "Will extract one frame every {} frames to get approximately {}.",
        interval, NUM_FRAMES_TO_EXTRACT
    );

    let mut current_frame_index = 0;
    let mut saved_frame_count = 0;
    let mut frame = Mat::default();

    // Loop through the video frames
    loop {
        if !cap.read(&mut frame)? || frame.empty() {
            break; // End of video
        }

        // Save frame if it's at the correct interval
        if current_frame_index % interval == 0 {
            let filename = format!("{}/frame_{:05}.jpg", OUTPUT_DIR, saved_frame_count);
            imgcodecs::imwrite(&filename, &frame, &core::Vector::new())?;
            saved_frame_count += 1;

            // Print progress
            print!(
                "\rSaved {}/{} frames...",
                saved_frame_count, NUM_FRAMES_TO_EXTRACT
            );
            io::stdout().flush()?;
        }

        current_frame_index += 1;

        if saved_frame_count >= NUM_FRAMES_TO_EXTRACT {
            break; // Stop when the desired number of frames is reached
        }
    }

    println!(
        "\nExtraction complete. {} frames saved in the '{}' folder.",
        saved_frame_count, OUTPUT_DIR
    );

    Ok(())
}