# Table Tennis Evaluation App

An interactive PySide6 application with PyQtGraph for visualizing table tennis match data from HDF5 files.

## Project Structure

```
data_plotter/
├── app.py              # Main application and UI coordination
├── plot_widgets.py     # Plot widgets (scatter plot, trajectory plots)
├── data_processor.py   # Data loading and processing logic
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Module Descriptions

- **app.py**: Main entry point and MainWindow class that coordinates the UI
- **plot_widgets.py**: Contains `FlightSegmentPlot` and `SpinSpeedScatterPlot` widgets
- **data_processor.py**: `DataProcessor` class for loading HDF5 files and extracting features

## Features

- **Load HDF5 Files**: Click a button to load match data from a directory containing HDF5 files
- **Player Type Selection**: Choose between Robot or Player data using a dropdown menu
- **View Level Selection**: Switch between Segment, Shot, and Rally views for different data granularity
- **Interactive Scatter Plot**: View spin magnitude vs speed magnitude for all flight segments
- **Clickable Points**: Click on any point in the scatter plot to view the corresponding flight segment trajectory
- **Flight Segment Visualization**: See X, Y, Z position plots over time (as markers) for selected flight segments
- **Metadata Display**: View match date, player, game name, game ID, and rally ID for each selected segment
- **Optimized Layout**: Left control panel with maximized plot area
- **Comprehensive Flight Data**: 3x3 grid showing position, velocity, and spin for all three axes

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the `ace_evaluation` package is in your Python path. The app expects to import from:
   - `ace_evaluation.utilities.data_classes`

## Usage

Run the application:

```bash
python app.py
```

### Using the App

1. **Load Data**: Click the "Load HDF5 File(s)" button and select a directory containing your HDF5 match data files
   - The app will automatically find and load all `.h5` files in the directory (excluding optitrack files)

2. **Select Player Type**: Use the dropdown menu to choose between "Robot" or "Player" data
   - This filters the scatter plot to show only shots from the selected player

3. **Select View Level**: Use the view selector dropdown in the right panel to choose:
   - **Segment**: Shows data for a single flight segment (between events)
   - **Shot**: Shows data for the entire shot (all segments combined)
   - **Rally**: Shows data for the complete rally

4. **View Summary**: After loading, the scatter plot will show spin magnitude vs speed magnitude for all flight segments
   - Each point represents the final state of a flight segment
   - The status bar shows the number of matches, games, and rallies loaded

5. **Explore Flight Segments**: Click on any point in the scatter plot to view:
   - **Metadata bar**: Shows match date, player name, game name/ID, and rally ID
   - **3x3 plot grid**:
     - **Row 1 (Position)**: X, Y, Z position over time (red markers)
     - **Row 2 (Velocity)**: X, Y, Z velocity over time (blue markers)
     - **Row 3 (Spin)**: X, Y, Z angular velocity over time (green markers)

## Data Structure

The app uses the data classes from `ace_evaluation.utilities.data_classes`:
- `MatchCollection`: Container for all matches
- `Match`: Contains games, date, player info
- `Game`: Contains rallies
- `Rally`: Contains shots
- `Shot`: Contains flight segments
- `FlightSegment`: Contains trajectory data (position, velocity, spin)

## Requirements

- PySide6 >= 6.5.0
- PyQtGraph >= 0.13.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- h5py >= 3.0.0

## Notes

- The scatter plot uses ground truth velocity data (`vx_gt200`, `vy_gt200`, `vz_gt200`) and angular velocity (`wx`, `wy`, `wz`)
- Only flight segments with valid data (no NaN values) are displayed
- The app filters out segments with less than 2 data points
- Time in trajectory plots is relative (starts from 0 for each segment)
- Trajectory plots use markers instead of lines for better visibility of individual data points
- Player filter: player=1 is robot, player=2 is human player
- The app starts in maximized/full-screen mode
- The left control panel is fixed at 250px width for easy access to controls
- Flight segment detail view shows 3x3 grid:
  - Columns: X, Y, Z components
  - Rows: Position (m), Velocity (m/s), Spin/Angular Velocity (rad/s)
- Velocity plots prefer ground truth data (`vx_gt200`) but fall back to APS data if unavailable
