# Flight Segment Detail View - 3x3 Grid Layout

## Visual Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Flight Segment Detail View                      │
├───────────────────────┬───────────────────────┬──────────────────────┤
│                       │                       │                      │
│    X Position (m)     │    Y Position (m)     │   Z Position (m)     │
│    🔴 Red markers     │    🔴 Red markers     │   🔴 Red markers     │
│                       │                       │                      │
├───────────────────────┼───────────────────────┼──────────────────────┤
│                       │                       │                      │
│   X Velocity (m/s)    │   Y Velocity (m/s)    │  Z Velocity (m/s)    │
│    🔵 Blue markers    │    🔵 Blue markers    │   🔵 Blue markers    │
│                       │                       │                      │
├───────────────────────┼───────────────────────┼──────────────────────┤
│                       │                       │                      │
│   X Spin (rad/s)      │   Y Spin (rad/s)      │  Z Spin (rad/s)      │
│    🟢 Green markers   │    🟢 Green markers   │   🟢 Green markers   │
│                       │                       │                      │
└───────────────────────┴───────────────────────┴──────────────────────┘
```

## Grid Organization

### Rows (Physical Properties):
1. **Position** (meters) - Ball position in 3D space
2. **Velocity** (meters/second) - Linear velocity of the ball
3. **Spin** (radians/second) - Angular velocity (spin) of the ball

### Columns (Coordinate Axes):
1. **X-axis** - First column
2. **Y-axis** - Second column  
3. **Z-axis** - Third column

### Color Coding:
- **🔴 Red**: Position data
- **🔵 Blue**: Velocity data
- **🟢 Green**: Spin/Angular velocity data

## Data Sources

### Position (Row 1):
- Uses `x_aps`, `y_aps`, `z_aps` from flight segment data

### Velocity (Row 2):
- Prefers ground truth: `vx_gt200`, `vy_gt200`, `vz_gt200`
- Falls back to APS: `vx_aps`, `vy_aps`, `vz_aps` if GT not available

### Spin (Row 3):
- Uses angular velocity: `wx`, `wy`, `wz` from flight segment data

## Visualization Style

- **Markers**: Small circles (size 5) instead of lines
- **Transparency**: 150/255 alpha for better visibility when overlapping
- **Grid**: Enabled on all plots for easier reading
- **Time**: X-axis shows relative time (starts at 0 for each segment)
- **Labels**: Each plot has appropriate axis labels with units

## Benefits

✅ **Comprehensive View**: See all 9 components of ball state at once  
✅ **Easy Comparison**: Compare behavior across different axes  
✅ **Physical Insight**: Position, velocity, and spin together reveal ball dynamics  
✅ **Organized Layout**: Logical grouping by property (rows) and axis (columns)  
✅ **Color Consistency**: Same color for same property type across all axes
