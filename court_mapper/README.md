# court_mapper

court_mapper is a Rust application for mapping and analyzing sports courts using computer vision. It leverages OpenCV for image and video processing, and uses Serde for handling JSON data. The project is designed to process court images and videos, identify field zones, and output structured data for further analysis.

## Features

- Court image and video processing using OpenCV
- Zone mapping and detection
- JSON-based configuration and output
- Extensible for different sports and court layouts

## Directory Structure

- `src/`: Rust source code
- `assets/`: Contains input files (`field.png`, `video.mp4`)
- `ddl/`: Required DLLs for OpenCV
- `field_zones.json`: Zone configuration file
- `Cargo.toml`, `Cargo.lock`: Rust project configuration

## Requirements

- Rust (edition 2024)
- Cargo
- OpenCV 4.1.1 DLLs (provided in `ddl/`)
- Windows OS

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd court_mapper
   ```
2. Ensure Rust and Cargo are installed:
   [Install Rust](https://www.rust-lang.org/tools/install)

3. Place required DLLs in `ddl/` and ensure they are accessible.

## Usage

1. Build the project:
   ```
   cargo build --release
   ```

2. Run the application:
   ```
   cargo run --release
   ```

3. Input files:
   - Place your court image in `assets/field.png`
   - Place your video in `assets/video.mp4`
   - Configure zones in `field_zones.json`

## Configuration

- `field_zones.json`: Define zones and their coordinates for mapping.

## Dependencies

- [opencv](https://crates.io/crates/opencv) (`0.95.1`)
- [serde](https://crates.io/crates/serde) (`1.0.219`)
- [serde_json](https://crates.io/crates/serde_json) (`1.0.142`)

## License

This project is licensed under the MIT License.

## Author

AndreaCarzeri

---

For more details, see the source code in `src/main.rs`.