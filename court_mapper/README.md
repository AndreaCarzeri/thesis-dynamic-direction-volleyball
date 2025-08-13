# Court Mapper

`court_mapper` is a Rust application designed for interactively defining and mapping polygonal zones on sports court images. It provides a graphical user interface built with OpenCV to allow users to draw multiple zones, assign custom metadata to each, and save the final layout to a structured JSON file for further analysis.

This tool is ideal for creating ground-truth data for sports analytics, enabling the spatial analysis of player or ball positions within specific areas of the court.

## Features

- **Multi-Zone Mapping:** Define up to 9 distinct polygonal zones on a single image.
- **Interactive Drawing:** Click to add vertices and create complex polygonal shapes.
- **Custom Metadata:** Assign a unique ID, Camera ID, Field ID, and Mode (e.g., "Defense", "Attack") to each zone.
- **Live Preview:** Real-time visual feedback with color-coding for active and inactive zones.
- **JSON Export:** Save the complete zone layout, including all points and metadata, to a `field_zones.json` file.
- **Video Frame Extraction:** Automatically extracts the first frame from a source video to use as a mapping canvas.

## Directory Structure

```
court_mapper/
├── src/
│   └── main.rs         # Main application logic
├── assets/
│   ├── video.mp4       # Source video to extract the mapping frame from
│   └── field.png       # The extracted frame used by the application (auto-generated)
├── field_zones.json    # Output file containing the saved zone data
├── Cargo.toml
└── Cargo.lock
```

## Requirements

- Rust (edition 2021 or newer)
- Cargo
- OpenCV (ensure system libraries are installed and accessible)
- A source video file (e.g., `assets/video.mp4`)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd court_mapper
    ```
2.  **Install Rust:** If you don't have it, install from [rust-lang.org](https://www.rust-lang.org/tools/install).
3.  **Install OpenCV:** Follow the setup guide for the [opencv-rust crate](https://github.com/opencv-rust/opencv-rust) for your operating system. This typically involves installing the OpenCV library and setting environment variables.

## Usage

1.  **Prepare a Source Video:** Place a video file (e.g., `video.mp4`) inside the `assets` directory. The application will automatically extract the first frame from this video and save it as `assets/field.png` to use as the mapping background.

2.  **Build and Run the Application:**
    ```bash
    cargo run --release
    ```
3.  An interactive window will open, displaying the extracted frame.

### Interactive Controls

The application is controlled entirely via the keyboard and mouse:

| Azione | Controllo | Descrizione |
| :--- | :--- | :--- |
| **Aggiungi Vertice** | **Click Sinistro** | Aggiunge un punto al perimetro della zona attualmente attiva. |
| **Seleziona Zona** | **Tasti `1` - `9`** | Seleziona la zona da modificare. La zona attiva è disegnata in **verde**, le altre in **rosso**. |
| **Chiudi Poligono** | **Tasto `d`** | Chiude la forma della zona attiva, collegando l'ultimo vertice al primo. |
| **Cicla Camera ID** | **Tasto `c`** | Modifica l'ID della telecamera (1-6) per la zona attiva. |
| **Cicla Campo ID** | **Tasto `f`** | Modifica l'ID del campo (0-1) per la zona attiva. |
| **Cicla Modalità** | **Tasto `m`** | Cambia la modalità ("Defense"/"Attack") per la zona attiva. |
| **Salva Layout** | **Tasto `s`** | Salva **tutte** le zone definite e i loro metadati nel file `field_zones.json`. |
| **Resetta Zona** | **Tasto `r`** | Cancella tutti i punti della **sola zona attiva**. |
| **Esci** | **Tasto `q`** | Chiude l'applicazione. |

The status of the currently selected zone (ID, Cam, Field, Mode) is displayed in the top-left corner of the window.

## Configuration Output (`field_zones.json`)

Pressing `s` will generate a `field_zones.json` file in the project's root directory. The file will contain an array of zone objects, structured as follows:

```json
[
  {
    "id": 1,
    "cam": 3,
    "mode": "Attack",
    "field": 0,
    "points": [
      { "x": 863, "y": 938 },
      { "x": 1514, "y": 927 },
      { "x": 1206, "y": 1779 }
    ],
    "is_closed": true
  },
  {
    "id": 2,
    "cam": 1,
    "mode": "Defense",
    "field": 1,
    "points": [
      { "x": 1906, "y": 924 },
      { "x": 2277, "y": 931 },
      { "x": 2566, "y": 1795 },
      { "x": 1918, "y": 1808 }
    ],
    "is_closed": true
  }
]
```

This JSON file can then be loaded by other applications for spatial analysis.

## Dependencies

-   [opencv](https://crates.io/crates/opencv)
-   [serde](https://crates.io/crates/serde)
-   [serde_json](https://crates.io/crates/serde_json)

## License

This project is licensed under the MIT License.

## Author

Andrea Carzeri