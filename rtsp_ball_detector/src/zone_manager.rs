// src/zone_manager.rs

use opencv::core as cv_core;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;

// --- STRUCTS FOR JSON DESERIALIZATION ---
// These structs exactly match the structure of your field_zones.json file.

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Zone {
    pub id: u32,
    pub field: u32,
    pub points: Vec<Point>,
    pub is_closed: bool,
}

// --- CORE FUNCTIONALITY ---

/// Loads zone definitions from a JSON file.
///
/// # Arguments
/// * `path` - The path to the field_zones.json file.
///
/// # Returns
/// A vector of `Zone` structs loaded from the file.
pub fn load_zones(path: &str) -> Result<Vec<Zone>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // First, deserialize into a temporary vector that may contain invalid zones
    let all_zones_from_file: Vec<Zone> = serde_json::from_reader(reader)?;

    println!("Loaded {} total definitions from {}. Validating...", all_zones_from_file.len(), path);

    // Filter the zones, keeping only the valid ones
    let valid_zones: Vec<Zone> = all_zones_from_file
        .into_iter() 
        .filter(|zone| {
            if !zone.is_closed {
                eprintln!(
                    "Warning: Skipping Zone ID {} because it is not marked as closed.",
                    zone.id
                );
                return false; 
            }
            
            if zone.points.len() < 3 {
                eprintln!(
                    "Warning: Skipping Zone ID {} because it has fewer than 3 points ({}).",
                    zone.id,
                    zone.points.len()
                );
                return false;
            }
            true
        })
        .collect();

    println!("Successfully loaded and validated {} zones.", valid_zones.len());
    Ok(valid_zones)
}

/// Determines which zone a given point belongs to.
///
/// Implements the Ray Casting algorithm (point in polygon test).
/// It checks how many times a ray starting from the point and going horizontally
/// to the right intersects with the edges of the polygon.
///
/// # Arguments
/// * `point` - The (x, y) coordinate to check, as an OpenCV Point.
/// * `zones` - A slice of `Zone` structs to check against.
///
/// # Returns
/// An `Option<u32>` containing the ID of the zone if the point is inside, otherwise `None`.
pub fn get_zone_for_point(point: cv_core::Point, zones: &[Zone]) -> Option<u32> {
    for zone in zones {

        let mut intersections = 0;
        let num_vertices = zone.points.len();

        for i in 0..num_vertices {
            let p1 = &zone.points[i];
            let p2 = &zone.points[(i + 1) % num_vertices]; // Next vertex, wraps around

            // Check if the point is vertically between the edge's y-coordinates
            if (p1.y > point.y) != (p2.y > point.y) {
                // Calculate the x-coordinate of the intersection with the horizontal ray
                let intersection_x = (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x;

                // If the intersection is to the right of the point, count it
                if point.x < intersection_x {
                    intersections += 1;
                }
            }
        }

        // If the number of intersections is odd, the point is inside the polygon
        if intersections % 2 == 1 {
            return Some(zone.id);
        }
    }

    // If the point is not in any zone
    None
}