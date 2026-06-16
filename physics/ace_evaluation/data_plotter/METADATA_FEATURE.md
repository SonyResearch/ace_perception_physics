# Flight Segment Metadata Display

## Feature Overview

When you click on a point in the scatter plot, the right panel now displays comprehensive metadata about the selected flight segment in an information bar above the 3x3 plot grid.

## Visual Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  Right Panel - Flight Segment Detail View                             │
├────────────────────────────────────────────────────────────────────────┤
│  Match Date: 2024-03-15 | Player: john_doe | Game: experiment_5      │
│  (ID: 2) | Rally ID: 7                                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                         3x3 Plot Grid                                  │
│                    (Position, Velocity, Spin)                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Metadata Information

The info bar displays:

1. **Match Date**: Date when the match was played
   - Format: YYYY-MM-DD
   - Source: `match.date`

2. **Player**: Name of the human player
   - Source: `match.player`
   - Extracted from match metadata

3. **Game Name**: Experiment/game identifier
   - Source: `game.game_name`
   - Usually contains policy/experiment information

4. **Game ID**: Numeric identifier for the game
   - Source: `game.game_id`
   - Zero-indexed within each match

5. **Rally ID**: Numeric identifier for the rally
   - Source: `rally.rally_id`
   - Zero-indexed within each game

## Implementation Details

### Data Flow:

1. **Data Processor** (`data_processor.py`):
   - `extract_spin_speed_data()` now returns 4 items instead of 3
   - Returns: `(speeds, spins, flight_segments, metadata_list)`
   - Each metadata dict contains: `match_date`, `match_player`, `game_name`, `game_id`, `rally_id`

2. **Scatter Plot Widget** (`plot_widgets.py`):
   - Stores `metadata_list` alongside `flight_segments`
   - Signal now emits both `FlightSegment` and `metadata dict`
   - `flight_segment_selected = Signal(object, object)`

3. **Flight Segment Plot** (`plot_widgets.py`):
   - New `info_label` at the top of the widget
   - Accepts metadata parameter in `plot_flight_segment()`
   - Formats and displays metadata as HTML

4. **Main Window** (`app.py`):
   - Updated to handle 4-tuple from data processor
   - Passes metadata to scatter plot
   - Passes metadata from signal to detail plot

### Styling:

```python
self.info_label.setStyleSheet(
    "QLabel { "
    "padding: 5px; "
    "background-color: #f0f0f0; "
    "border: 1px solid #ccc; "
    "}"
)
```

- Light gray background (#f0f0f0)
- Subtle border (#ccc)
- 5px padding for readability
- Word wrapping enabled
- HTML formatted text with bold labels

## Benefits

✅ **Context awareness**: Know exactly which match/game/rally you're viewing  
✅ **Traceability**: Easy to reference specific segments in analysis  
✅ **Debugging**: Quickly identify data from specific games or dates  
✅ **Organization**: Better understanding of data provenance  
✅ **Professional appearance**: Clean, organized information display

## Example Display

```
Match Date: 2024-11-15 | Player: alice_smith | Game: tt_ace_vs_alice_match_3 (ID: 1) | Rally ID: 12
```

The metadata appears in a clean, single-line format with clear separators (|) between fields and bold labels for easy scanning.
