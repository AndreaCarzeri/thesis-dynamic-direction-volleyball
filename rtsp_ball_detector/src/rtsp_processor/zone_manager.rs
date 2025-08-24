use anyhow::Result;
use opencv::core as cv_core;
use opencv::core::Point_;
use opencv::{
    core::{self, Mat},
    imgproc,
};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;
use std::fs::File;
use std::io::BufReader;

/// Represents the tactical mode of a zone.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum Mode {
    Attack,
    Defense,
}

/// Represents a 2D coordinate point.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

/// Defines a polygonal area on the field with associated properties.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Zone {
    pub id: i32,
    pub field: i32,
    pub points: Vec<Point>,
    pub is_closed: bool,
    pub cam: u32,
    pub mode: Mode,
}

/// Manages all zones, the active camera, and the logic for switching between them.
pub struct ZoneManager {
    zones: Vec<Zone>,
    active_zone: Zone,
    last_switch_time: std::time::Instant,
    change_threshold: std::time::Duration,
    last_ball_seen_time: std::time::Instant,
    print: bool,
}

const OUTSIDE_CAM_ID: u32 = 6;
const NO_BALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

impl ZoneManager {
    /// Returns a reference to the currently active zone.
    pub fn active_zone(&self) -> &Zone {
        &self.active_zone
    }

    /// Updates the active camera based on ball position and predefined switching rules.
    pub fn update_active_cam(&mut self, ball_pos: Option<Point_<i32>>) {
        if let Some(ball_pos) = ball_pos {
            self.last_ball_seen_time = std::time::Instant::now();

            let Some(current_zone) = get_zone_for_point(ball_pos, &self.zones) else {
                if self.active_zone.id != -1 {
                    self.print_change_cam("Valid transition to", &self.active_zone);
                    self.active_zone = self.zones.last().unwrap().clone();
                    self.last_switch_time = std::time::Instant::now();
                }
                return;
            };

            if self.last_switch_time.elapsed() >= self.change_threshold {
                if self.print {
                    println!(
                        "CAMERA SWITCH: Change threshold met {:?} >= {:?}. Considering switch to zone {}.",
                        self.last_switch_time.elapsed(),
                        self.change_threshold,
                        current_zone.id
                    );
                }

                if current_zone.field != self.active_zone.field {
                    if current_zone.mode != Mode::Attack {
                        self.print_change_cam("Valid transition to", current_zone);

                        self.active_zone = current_zone.clone();
                        self.last_switch_time = std::time::Instant::now();
                    }
                } else if current_zone.field == self.active_zone.field
                    && self.active_zone.mode == Mode::Defense
                    && current_zone.mode == Mode::Attack
                {
                    self.print_change_cam("Valid transition to", current_zone);
                    self.active_zone = current_zone.clone();
                    self.last_switch_time = std::time::Instant::now();
                }
            } else {
                self.print_change_cam(
                    "Change threshold not met. Staying on current",
                    &self.active_zone,
                );
            }
        } else if self.last_ball_seen_time.elapsed() > NO_BALL_TIMEOUT && self.active_zone.id != -1
        {
            self.print_change_cam(&format!("NO BALL TIMEOUT: Ball not seen for > {:.1}s. Switching to", NO_BALL_TIMEOUT.as_secs_f32()), &self.zones.last().unwrap());
            self.active_zone = self.zones.last().unwrap().clone();
            self.last_switch_time = std::time::Instant::now();
        }
    }

    fn print_change_cam(&self, message: &str, new_zone: &Zone) {
        if self.print {
            println!(
                "\n>>> ZONE MANAGER: {} zone {}, active cam: {} <<<\n",
                message, new_zone.id, new_zone.cam
            );
        }
    }

    /// Draws a semi-transparent overlay of all defined zones onto an image frame.
    pub fn draw_zones_overlay(&self, frame: &mut Mat) -> Result<()> {
        let alpha = 0.3;
        let mut overlay = frame.clone();

        for zone in &self.zones {
            if zone.points.len() > 2 {
                let points: Vec<core::Point> = zone
                    .points
                    .iter()
                    .map(|p| core::Point::new(p.x, p.y))
                    .collect();
                let points_vec = opencv::core::Vector::from_slice(&points);

                imgproc::fill_poly(
                    &mut overlay,
                    &points_vec,
                    generate_color_scalar(zone.id as u32),
                    imgproc::LINE_8,
                    0,
                    core::Point::default(),
                )?;
            }
        }

        let frame_clone = frame.clone();
        core::add_weighted(&frame_clone, 1.0 - alpha, &overlay, alpha, 0.0, frame, -1)?;

        Ok(())
    }
}

/// Loads and validates zone definitions from a JSON file into a ZoneManager.
pub fn load_zones(path: &str, change_threshold: f32, print: bool) -> Result<ZoneManager> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let all_zones_from_file: Vec<Zone> = serde_json::from_reader(reader)?;

    let mut valid_zones: Vec<Zone> = all_zones_from_file
        .into_iter()
        .filter(|zone| zone.is_closed && zone.points.len() >= 3)
        .collect();

    if valid_zones.is_empty() {
        anyhow::bail!("No valid zones were loaded from the file.");
    }
    valid_zones.push(Zone {
        id: -1,
        field: -1,
        points: vec![],
        is_closed: true,
        cam: OUTSIDE_CAM_ID,
        mode: Mode::Defense,
    });

    println!(
        "Successfully loaded and validated {} zones.",
        valid_zones.len()
    );

    let zone_manager = ZoneManager {
        zones: valid_zones.clone(),
        active_zone: valid_zones.last().unwrap().clone(),
        last_switch_time: std::time::Instant::now(),
        change_threshold: std::time::Duration::from_secs_f32(change_threshold),
        last_ball_seen_time: std::time::Instant::now(),
        print,
    };
    Ok(zone_manager)
}

/// Determines which zone contains a given point using the ray casting algorithm.
pub fn get_zone_for_point(point: cv_core::Point, zones: &[Zone]) -> Option<&Zone> {
    for zone in zones {
        let mut intersections = 0;
        let num_vertices = zone.points.len();

        for i in 0..num_vertices {
            let p1 = &zone.points[i];
            let p2 = &zone.points[(i + 1) % num_vertices];

            if (p1.y > point.y) != (p2.y > point.y) {
                let intersection_x = (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x;
                if point.x < intersection_x {
                    intersections += 1;
                }
            }
        }

        if intersections % 2 == 1 {
            return Some(zone);
        }
    }
    None
}

/// Generates a visually distinct color for a given integer ID.
pub fn generate_color_scalar(n: u32) -> core::Scalar {
    const GOLDEN_RATIO_CONJUGATE: f64 = 0.618033988749895;
    const SATURATION: f64 = 0.8;
    const LIGHTNESS: f64 = 0.65;

    let hue = (n as f64 * GOLDEN_RATIO_CONJUGATE) % 1.0;
    let (r_u8, g_u8, b_u8) = hsl_to_rgb(hue, SATURATION, LIGHTNESS);

    core::Scalar::new(r_u8 as f64, g_u8 as f64, b_u8 as f64, 255.0)
}

/// Converts a color from HSL to RGB color space.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    if s == 0.0 {
        let val = (l * 255.0).round() as u8;
        return (val, val, val);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
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

/// A helper function for HSL to RGB conversion.
fn hue_to_rgb_component(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }

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
