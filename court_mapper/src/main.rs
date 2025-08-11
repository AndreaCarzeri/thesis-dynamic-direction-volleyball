use std::collections::HashMap;
use opencv::{
    core, highgui, imgproc,
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};

// A struct to make our points serializable to JSON
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializablePoint {
    x: i32,
    y: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Zone {
    id: u32,
    points: Vec<SerializablePoint>,
    is_closed: bool,
}

#[derive(Debug, Clone)]
struct AppState {
    // Maps Zone ID to the list of points (OpenCV Point format)
    zones: HashMap<u32, Vec<core::Point>>,
    // Maps Zone ID to its closed status
    closed_status: HashMap<u32, bool>,
    // The currently active zone being edited
    active_zone_id: u32,
}

fn extract_first_frame(video_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Extracting first frame from: {}", video_path);

    let mut cap = opencv::videoio::VideoCapture::from_file(video_path, opencv::videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        print!("Error: Could not open video file at path: {}", video_path);
        return Err(format!("Could not open video file at path: {}", video_path).into());
    }

    let mut frame = Mat::default();
    if !cap.read(&mut frame)? || frame.empty() {
        print!("Error: Could not read the first frame. File might be empty/corrupted.");
        return Err("Could not read the first frame. File might be empty/corrupted.".into());
    }

    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let params = opencv::core::Vector::new();
    opencv::imgcodecs::imwrite(output_path, &frame, &params)?;

    println!("First frame successfully saved to: {}", output_path);

    cap.release()?;

    Ok(())
}

// Main function
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let video_for_mapping = "assets/video.mp4";
    let image_path = "assets/field.png";
    // Uncomment the next line if you need to extract the first frame
    // extract_first_frame(video_for_mapping, image_path)?;

    let source_image = opencv::imgcodecs::imread(image_path, opencv::imgcodecs::IMREAD_COLOR)?;
    if source_image.empty() {
        panic!("Could not open or find the image at: {}", image_path);
    }

    let window_name = "Polygon Zone Mapper";
    highgui::named_window(window_name, highgui::WINDOW_NORMAL)?;

    // --- SHARED STATE ---
    let app_state = Arc::new(Mutex::new(AppState {
        zones: HashMap::new(),
        closed_status: HashMap::new(),
        active_zone_id: 1,
    }));
    let state_clone = Arc::clone(&app_state);

    // Mouse callback to add points to the active zone
    highgui::set_mouse_callback(window_name, Some(Box::new(move |event, x, y, _flags| {
        if event == highgui::EVENT_LBUTTONDOWN {
            let mut state_guard = state_clone.lock().unwrap();
            let active_id = state_guard.active_zone_id;

            // Add point to the currently active zone
            state_guard.zones.entry(active_id).or_insert_with(Vec::new).push(core::Point::new(x, y));
        }
    })))?;

    println!("--- Polygon Zone Mapper ---");
    println!("Select zone: 1-9 (default is Zone 1).");
    println!("Click to add points.");
    println!("'d' (done) -> Close the polygon.");
    println!("'s' (save) -> Save all zones to 'field_zones.json'.");
    println!("'r' (reset) -> Clear the active zone.");
    println!("'q' (quit) -> Exit the application.");

    loop {
        let mut display_image = source_image.clone();
        let mut state_guard = app_state.lock().unwrap().clone();

        let current_active_id = state_guard.active_zone_id;

        // Draw all zones
        for (zone_id, points) in state_guard.zones.iter() {
            if points.is_empty() { continue; }

            // Determine color (green for active, red for others)
            let color = if *zone_id == current_active_id {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
            } else {
                core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
            };

            let is_closed = *state_guard.closed_status.get(zone_id).unwrap_or(&false);

            // Draw points
            for point in points {
                imgproc::circle(&mut display_image, *point, 8, color, -1, imgproc::LINE_AA, 0)?;
            }

            // Draw lines
            let points_vec = opencv::core::Vector::from_slice(&points);
            imgproc::polylines(&mut display_image, &points_vec, is_closed, color, 3, imgproc::LINE_AA, 0)?;
        }

        highgui::imshow(window_name, &display_image)?;

        let key = highgui::wait_key(30)?;
        match key {
            -1 => {}, // No key pressed
            113 => break, // 'q' to quit

            // 'r' to reset (clear) the active zone
            114 => {
                let mut state_guard = app_state.lock().unwrap();
                state_guard.zones.insert(current_active_id, Vec::new());
                state_guard.closed_status.insert(current_active_id, false);
                println!("Zone {} reset.", current_active_id);
            },

            // 'd' to close the active polygon
            100 => {
                let mut state_guard = app_state.lock().unwrap();
                let points = state_guard.zones.get(&current_active_id).unwrap();
                if points.len() > 2 {
                    state_guard.closed_status.insert(current_active_id, true);
                    println!("Zone {} closed. Press 's' to save.", current_active_id);
                } else {
                    println!("Need at least 3 points to close Zone {}.", current_active_id);
                }
            },

            // 's' to save all zones
            115 => {
                let mut state_guard = app_state.lock().unwrap();
                let mut all_zones = Vec::new();
                for (id, points) in state_guard.zones.iter() {
                    let is_closed = *state_guard.closed_status.get(id).unwrap_or(&false);
                    let serializable_points: Vec<SerializablePoint> = points.iter()
                        .map(|p| SerializablePoint { x: p.x, y: p.y })
                        .collect();

                    all_zones.push(Zone {
                        id: *id,
                        points: serializable_points,
                        is_closed,
                    });
                }

                let mut file = File::create("field_zones.json")?;
                let json_string = serde_json::to_string_pretty(&all_zones)?;
                file.write_all(json_string.as_bytes())?;
                println!("All zones saved to field_zones.json!");
            },

            // 1-9 to select active zone
            49..=57 => {
                let mut state_guard = app_state.lock().unwrap();
                let selected_id = (key - 48) as u32; // ASCII values for 1-9
                state_guard.active_zone_id = selected_id;
                println!("Editing Zone {}. Click to add points.", selected_id);
            },
            _ => {} // Ignore other keys
        }
    }

    Ok(())
}