mod structs;

use opencv::{
    core::{self, Mat},
    highgui, imgproc,
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use structs::{AppState, SerializablePoint, Zone, ZoneMetadata};

/// Extracts the first valid frame from a video and saves it as a PNG image.
fn extract_first_frame(
    video_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Extracting first frame from: {}", video_path);

    let mut cap = opencv::videoio::VideoCapture::from_file(video_path, opencv::videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(format!("Could not open video file at path: {}", video_path).into());
    }

    let mut frame = Mat::default();
    if !cap.read(&mut frame)? || frame.empty() {
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

fn load_state_from_file(path: &str) -> Result<AppState, Box<dyn std::error::Error>> {
    // Open and read the file
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);

    // Deserialize the JSON into a vector of Zone structs
    let zones_from_file: Vec<Zone> = serde_json::from_reader(reader)?;

    // Create new HashMaps to build the AppState
    let mut zones = HashMap::new();
    let mut closed_status = HashMap::new();
    let mut metadata = HashMap::new();

    // Iterate over the loaded zones and populate the HashMaps
    for zone_data in zones_from_file {
        let points: Vec<core::Point> = zone_data.points.iter()
            .map(|p| core::Point::new(p.x, p.y))
            .collect();

        zones.insert(zone_data.id+1, points);
        closed_status.insert(zone_data.id, zone_data.is_closed);
        metadata.insert(zone_data.id, ZoneMetadata {
            cam: zone_data.cam,
            mode: zone_data.mode,
            field: zone_data.field,
        });
    }

    // Return the newly constructed AppState
    Ok(AppState {
        zones,
        closed_status,
        metadata,
        active_zone_id: 1, // Default to editing zone 1 at startup
    })
}

/// Main application function.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let video_for_mapping = "assets/video.mp4";
    let image_path = "assets/field.png";
    let zones_json_path = "field_zones.json";
    // Uncomment the next line if you need to extract the first frame
    // extract_first_frame(video_for_mapping, image_path)?;

    let source_image = opencv::imgcodecs::imread(image_path, opencv::imgcodecs::IMREAD_COLOR)?;
    if source_image.empty() {
        panic!("Could not open or find the image at: {}", image_path);
    }

    let window_name = "Polygon Zone Mapper";
    highgui::named_window(window_name, highgui::WINDOW_NORMAL)?;
    let initial_state = match load_state_from_file(zones_json_path) {
        Ok(state) => {
            println!("Successfully loaded existing state from {}", zones_json_path);
            state
        },
        Err(_) => {
            println!("No existing state file found. Starting with a fresh session.");
            AppState {
                zones: HashMap::new(),
                closed_status: HashMap::new(),
                metadata: HashMap::new(),
                active_zone_id: 1,
            }
        }
    };

    let app_state = Arc::new(Mutex::new(initial_state));

    // --- SHARED STATE ---
    let state_clone = Arc::clone(&app_state);

    // Mouse callback to add points to the active zone
    highgui::set_mouse_callback(
        window_name,
        Some(Box::new(move |event, x, y, _flags| {
            if event == highgui::EVENT_LBUTTONDOWN {
                let mut state_guard = state_clone.lock().unwrap();
                let active_id = state_guard.active_zone_id;
                state_guard
                    .zones
                    .entry(active_id)
                    .or_insert_with(Vec::new)
                    .push(core::Point::new(x, y));
            }
        })),
    )?;

    println!("--- Polygon Zone Mapper ---");
    println!("Select zone: 1-9.");
    println!("'c' -> Cycle Camera ID (1-6).");
    println!("'f' -> Cycle Field ID (0-1).");
    println!("'m' -> Cycle Mode (Defense/Attack).");
    println!("'d' -> Close polygon. 's' -> Save all zones. 'r' -> Reset active zone. 'q' -> Quit.");

    loop {
        let mut display_image = source_image.clone();

        // Lock, clone the state, and immediately release the lock
        let current_state = app_state.lock().unwrap().clone();
        let current_active_id = current_state.active_zone_id;

        // Draw all zones, highlighting the active one
        for (zone_id, points) in &current_state.zones {
            if points.is_empty() {
                continue;
            }

            let color = if *zone_id == current_active_id {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for active
            } else {
                core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red for inactive
            };
            let is_closed = *current_state.closed_status.get(zone_id).unwrap_or(&false);

            for point in points {
                imgproc::circle(
                    &mut display_image,
                    *point,
                    8,
                    color,
                    -1,
                    imgproc::LINE_AA,
                    0,
                )?;
            }
            let points_vec = opencv::core::Vector::from_slice(points);
            imgproc::polylines(
                &mut display_image,
                &points_vec,
                is_closed,
                color,
                3,
                imgproc::LINE_AA,
                0,
            )?;
        }

        // Display the current status of the active zone
        let active_meta = current_state
            .metadata
            .get(&current_active_id)
            .cloned()
            .unwrap_or_default();
        let status_text = format!(
            "Editing Zone: {} | Cam: {} | Field: {} | Mode: {}",
            current_active_id, active_meta.cam, active_meta.field, active_meta.mode
        );
        imgproc::put_text(
            &mut display_image,
            &status_text,
            core::Point::new(10, 30),
            32,
            1.0,
            core::Scalar::new(255.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow(window_name, &display_image)?;

        let key = highgui::wait_key(30)?;
        match key {
            -1 => {}      // No key pressed
            113 => break, // 'q' to quit

            114 => {
                // 'r' to reset active zone
                let mut state_guard = app_state.lock().unwrap();
                let active_id = state_guard.active_zone_id;
                state_guard
                    .zones
                    .insert(active_id, Vec::new());
                state_guard
                    .closed_status
                    .insert(active_id, false);
                state_guard
                    .metadata
                    .insert(active_id, ZoneMetadata::default());
                println!("Zone {} reset.", state_guard.active_zone_id);
            }

            100 => { // 'd' to close the active polygon
                let mut state_guard = app_state.lock().unwrap();
                let active_id = state_guard.active_zone_id;
                let has_enough_points = state_guard.zones.get(&active_id)
                    .map_or(false, |points| points.len() > 2);

                if has_enough_points {
                    state_guard.closed_status.insert(active_id, true);
                    println!("Zone {} closed.", active_id);
                } else {
                    println!("Need at least 3 points to close Zone {}.", active_id);
                }
            }

            115 => {
                // 's' to save all zones
                let state_guard = app_state.lock().unwrap();
                let mut all_zones_to_save = Vec::new();
                for (id, points) in &state_guard.zones {
                    let meta = state_guard.metadata.get(id).cloned().unwrap_or_default();
                    let is_closed = *state_guard.closed_status.get(id).unwrap_or(&false);
                    let serializable_points: Vec<SerializablePoint> = points
                        .iter()
                        .map(|p| SerializablePoint { x: p.x, y: p.y })
                        .collect();
                    all_zones_to_save.push(Zone {
                        id: *id-1,
                        cam: meta.cam,
                        mode: meta.mode.clone(),
                        field: meta.field,
                        points: serializable_points,
                        is_closed,
                    });
                }
                let mut file = File::create(zones_json_path)?;
                let json_string = serde_json::to_string_pretty(&all_zones_to_save)?;
                file.write_all(json_string.as_bytes())?;
                println!("All zones saved to {}!", zones_json_path);
            }

            49..=57 => {
                // 1-9 to select active zone
                let mut state_guard = app_state.lock().unwrap();
                let selected_id = (key - 48) as u32;
                state_guard.active_zone_id = selected_id;
                println!("Editing Zone {}.", selected_id);
            }

            99 => { // 'c' to cycle camera ID
                let mut state_guard = app_state.lock().unwrap();
                let active_id = state_guard.active_zone_id;
                let meta = state_guard.metadata.entry(active_id).or_insert_with(Default::default);
                meta.cam = (meta.cam % 13) + 1;
                println!("Zone {} Camera set to: {}", active_id, meta.cam);
            },

            102 => { // 'f' to cycle field ID
                let mut state_guard = app_state.lock().unwrap();
                let active_id = state_guard.active_zone_id;
                let meta = state_guard.metadata.entry(active_id).or_insert_with(Default::default);
                meta.field = (meta.field + 1) % 2; // Cycle 0 -> 1 -> 0
                println!("Zone {} Field set to: {}", active_id, meta.field);
            },

            109 => { // 'm' to cycle mode
                let mut state_guard = app_state.lock().unwrap();
                let active_id = state_guard.active_zone_id;
                let meta = state_guard.metadata.entry(active_id).or_insert_with(Default::default);
                meta.mode = if meta.mode == "Defense" { "Attack".to_string() } else { "Defense".to_string() };
                println!("Zone {} Mode set to: {}", active_id, meta.mode);
            },
            _ => {}
        }
    }

    Ok(())
}
