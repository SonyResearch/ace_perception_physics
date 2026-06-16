# View Level Selector Feature

## Overview

The right panel now includes a **View Level Selector** dropdown that allows you to switch between three different levels of data granularity when viewing ball trajectories:

1. **Segment View** (default)
2. **Shot View**
3. **Rally View**

## Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Right Panel - Flight Segment Detail View                   │
├──────────────────────────────────────────┬──────────────────┤
│  Match Info: Date | Player | Game | Rally│  View: [Segment▼]│
├──────────────────────────────────────────┴──────────────────┤
│                                                              │
│                    3x3 Plot Grid                             │
│              (Position, Velocity, Spin)                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## View Levels Explained

### 🔹 **Segment View**
- **Scope**: Single flight segment (one part of a shot)
- **Data**: From one event to the next (e.g., from racket hit to table bounce)
- **Use Case**: Detailed analysis of ball behavior between specific events
- **Example**: Ball trajectory from robot hit to opponent's table bounce

**What is a Segment?**
A segment is one continuous portion of the ball's flight between events:
- Serve segments: 3 segments (racket → table → net → opponent table)
- Regular shots: typically 2 segments (racket → net/table → opponent)

### 🔹 **Shot View**
- **Scope**: Complete shot (all segments combined)
- **Data**: From when one player hits until the other player hits
- **Use Case**: Analyzing the full trajectory of a single shot
- **Example**: Complete ball path from robot hit to player hit

**What is a Shot?**
A shot includes all flight segments from one player's contact to the next:
- Start: Player makes contact with ball
- End: Opponent makes contact (or rally ends)
- Contains: All bounces, net contacts, and flight paths in between

### 🔹 **Rally View**
- **Scope**: Entire rally
- **Data**: All shots and segments from rally start to end
- **Use Case**: Understanding overall rally patterns and full exchange
- **Example**: Complete rally from serve to winning/losing point

**What is a Rally?**
A rally is the complete exchange:
- Start: Serve
- Middle: Back-and-forth shots
- End: Point scored (or error)
- Contains: All shots from both players

## How It Works

### Data Hierarchy
```
Rally
├── Shot 1 (Player A)
│   ├── Segment 1 (racket → table)
│   └── Segment 2 (table → opponent)
├── Shot 2 (Player B)
│   ├── Segment 1 (racket → table)
│   └── Segment 2 (table → opponent)
└── ...more shots...
```

### Switching Views

1. Click on any point in the scatter plot (left panel)
2. The right panel displays the segment's data
3. Use the "View" dropdown to switch between levels
4. The plots update automatically to show the selected scope

### Data Continuity

When you switch views:
- **Segment → Shot**: Plots expand to show all segments in that shot
- **Shot → Rally**: Plots expand to show all shots in that rally
- **Rally → Segment**: Plots return to just the clicked segment

## Technical Implementation

### Data Storage
Each clicked point stores:
- `flight_segment`: The specific segment clicked
- `shot_data`: Combined DataFrame of all segments in the shot
- `rally_data`: Complete rally DataFrame

### View Rendering
```python
if view_type == "Segment":
    data = flight_segment.data
elif view_type == "Shot":
    data = shot_data  # Concatenated segments
elif view_type == "Rally":
    data = rally_data  # Full rally DataFrame
```

### Data Combination
- **Shot data**: `pd.concat([seg.data for seg in shot.flight_segments])`
- **Rally data**: Original `rally.rally` DataFrame from data classes

## Use Cases

### 🎯 **Segment View**
Best for:
- Analyzing ball behavior after specific events
- Studying bounce characteristics
- Understanding spin decay during flight
- Examining velocity changes in short intervals

### 🎯 **Shot View**
Best for:
- Comparing complete shot trajectories
- Understanding shot strategy
- Analyzing serve effectiveness
- Studying shot-to-shot patterns

### 🎯 **Rally View**
Best for:
- Overall rally analysis
- Tracking rally momentum
- Understanding point development
- Comparing player strategies over exchanges

## Visual Differences

### Segment View
```
Time: 0.0s ────→ 0.2s
      |━━━━━━━━|
      Hit    Bounce
```

### Shot View
```
Time: 0.0s ─────────→ 0.8s
      |━━━━|━━━━|━━━━|
      Hit  Bounce Net  Hit
```

### Rally View
```
Time: 0.0s ──────────────────→ 5.0s
      |━━━━━|━━━━━|━━━━━|━━━━━|
      Serve  Shot2 Shot3 Shot4
```

## Benefits

✅ **Flexibility**: View data at the granularity you need  
✅ **Context**: Understand where a segment fits in the larger picture  
✅ **Comparison**: Switch views to see different perspectives  
✅ **Analysis**: Choose the right level for your analysis needs  
✅ **Seamless**: Instant switching without re-clicking points  

## Notes

- The view selector remembers your last selection
- Switching views uses cached data (no reloading)
- Metadata bar remains constant across view changes
- All 9 plots (position, velocity, spin × X, Y, Z) update together
- Time axis adjusts automatically to show full scope
