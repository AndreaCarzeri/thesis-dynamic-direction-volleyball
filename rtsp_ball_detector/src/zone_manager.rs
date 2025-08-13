// src/zone_manager.rs

use std::cmp::PartialEq;
use opencv::core as cv_core;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;
use opencv::{core::{self, Mat}, highgui, imgproc, prelude::*, videoio::{self, VideoCapture}};
use opencv::core::Point_;
use crate::zone_manager;
// --- STRUCTS FOR JSON DESERIALIZATION ---
// These structs exactly match the structure of your field_zones.json file.
#[derive(Serialize, Deserialize, Clone, Debug)]
enum Mode {
    Attack,
    Defense
}

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
    pub cam: u32,
    pub mode: Mode
}

pub struct ZoneManager {
    zones: Vec<Zone>,
    threshold: f32, //seconds to change cam
    old_zone: Zone,
}

impl PartialEq for Mode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Mode::Attack, Mode::Attack) => true,
            (Mode::Defense, Mode::Defense) => true,
            _ => false,
        }
    }
}

impl ZoneManager {
    /// Returns a reference to the zones.
    pub fn zones(&self) -> &[Zone] {
        &self.zones
    }

    /// Returns the threshold for changing camera.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns the current field ID.
    pub fn old_zone(&self) -> Zone {
        self.old_zone.clone()
    }

    pub fn get_cam(&mut self, ball_pos: Point_<i32>) -> u32 {
        if let Some(actual_zone) = get_zone_for_point(ball_pos, &self.zones) {
            if actual_zone.field != self.old_zone.field{
                if actual_zone.mode == Mode::Attack {
                    println!("Ball is in zone {:?}, no changing camera", actual_zone);
                }else{
                    //TODO check threshold
                    self.old_zone = actual_zone.clone();
                    //println!("Ball is in zone {:?}, changing camera to {}", actual_zone, self.old_zone.cam);
                }
                self.old_zone.cam
            }else{
                if actual_zone.mode == Mode::Attack && self.old_zone.mode == Mode::Defense {
                    //TODO check threshold
                    self.old_zone = actual_zone.clone();
                    //println!("Ball is in zone {:?}, changing camera to {}", actual_zone, self.old_zone.cam);
                } else {
                    //println!("Ball is in zone {:?}, but no camera change needed.", actual_zone);
                }
                self.old_zone.cam
            }
        }else{
            //eprintln!("Ball position ({}, {}) is outside any defined zone.", ball_pos.x, ball_pos.y);
            return 6; // Return 0 if the point is not in any zone
        }
    }
    
    pub fn draw_zones_overlay(&self, frame: &mut Mat) -> Result<()> {
        // Scegliamo un colore semi-trasparente per l'overlay
        let alpha = 0.3; // Trasparenza

        // Crea un'immagine temporanea su cui disegnare i poligoni pieni
        let mut overlay = frame.clone();

        for zone in &self.zones {
            if zone.points.len() > 2 {
                let points: Vec<core::Point> = zone.points.iter()
                    .map(|p| core::Point::new(p.x, p.y))
                    .collect();
                let points_vec = opencv::core::Vector::from_slice(&points);

                // Disegna il poligono pieno sull'immagine temporanea
                imgproc::fill_poly(&mut overlay, &points_vec, generate_color_scalar(zone.id), imgproc::LINE_8, 0, core::Point::default())?;
            }
        }

        let frame_clone = frame.clone();
        core::add_weighted(&frame_clone, 1.0 - alpha, &overlay, alpha, 0.0, frame, -1)?;

        Ok(())
    }
}

// --- CORE FUNCTIONALITY ---

/// Loads zone definitions from a JSON file.
///
/// # Arguments
/// * `path` - The path to the field_zones.json file.
///
/// # Returns
/// A vector of `Zone` structs loaded from the file.
pub fn load_zones(path: &str) -> Result<ZoneManager> {
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
    let zone_manager =ZoneManager {
        zones: valid_zones.clone(),
        threshold: 1.0,
        old_zone: valid_zones[0].clone(),
    };
    Ok(zone_manager)
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
pub fn get_zone_for_point(point: cv_core::Point, zones: &[Zone]) -> Option<&Zone> {
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
            return Some(zone);
        }
    }

    // If the point is not in any zone
    None
}

pub fn generate_color_scalar(n: u32) -> core::Scalar {
    const GOLDEN_RATIO_CONJUGATE: f64 = 0.618033988749895;
    const SATURATION: f64 = 0.8;
    const LIGHTNESS: f64 = 0.65;

    // 1. Calcola la tonalità per avere colori distinti
    let hue = (n as f64 * GOLDEN_RATIO_CONJUGATE) % 1.0;

    // 2. Converte da HSL a RGB (valori u8 da 0 a 255)
    let (r_u8, g_u8, b_u8) = hsl_to_rgb(hue, SATURATION, LIGHTNESS);

    // 3. Converte i valori u8 in f64 e crea l'oggetto Scalar
    // Nota: L'ordine è R, G, B, A come da tua ultima richiesta.
    core::Scalar::new(
        r_u8 as f64,
        g_u8 as f64,
        b_u8 as f64,
        255.0, // Alfa impostato a 255.0 (opaco), più comune di 0.0 (trasparente)
    )
}

/// Converts an HSL color value to RGB. All inputs (h, s, l) are in [0.0, 1.0].
/// Returns (r, g, b) tuple with values in [0, 255].
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    if s == 0.0 {
        let val = (l * 255.0).round() as u8;
        return (val, val, val); // Achromatic (gray)
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    let r = hue_to_rgb_component(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb_component(p, q, h);
    let b = hue_to_rgb_component(p, q, h - 1.0 / 3.0);

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

/// Helper function for hsl_to_rgb.
fn hue_to_rgb_component(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }

    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}