# Pro-Player EVS / APS Labeling Pipeline

End-to-end documentation of the data preparation pipeline that turns raw
event-camera (EVS) recordings + APS-derived ball trajectories into the
time-aligned `.h5` event/label pairs consumed by training and evaluation in
this repo.

---

## 1. Quick start

```bash
# 0. Build / source the workspace once (only needed if scripts use ROS bits)
colcon build --symlink-install --cmake-args --packages-up-to data_generation 
source ~/evs/install/setup.bash

# 1. Extract APS shutter triggers (one triggers.txt per evs/ folder)
python src/data_generation/tools/extract_triggers.py \
    --root_dir /path/to/<root_folder>

# 2. Generate per-rally CSVs and 2D label H5s from labels/*.pt
python src/data_generation/label_pro_player_data.py \
    --root_dir /path/to/<recording_folder> \
    --cam_name evs00050026

# 3. Slice events into per-rally H5 files using triggers + label CSVs
python src/data_generation/label_pro_player_data_events.py \
    --root_dir /path/to/<recording_folder> \
    --cam_name evs00050026

# 4. (Optional) sanity-check with a video (or omit _zoomed for the full view.)
python src/data_generation/convert_h5_to_video_zoomed.py \
    --root_dir /path/to/<recording_folder>/h5/evs00050026 \
    --target_dir /path/to/output/videos \
    --accumulation_time 5 --debug

```

For batch processing across many recordings & cameras, see
[run_pro_player_label_scripts.sh](run_pro_player_label_scripts.sh) and
[run_pro_player_label_scripts_all.sh](run_pro_player_label_scripts_all.sh).

---

## 2. Input data layout

The pipeline operates on **one recording folder** at a time. A "recording"
is one rally set / match / session with synchronized APS + EVS capture.

```
<recording_folder>/                       # passed as --root_dir
├── <calibration>.yaml                    # exactly ONE; multi-cam intrinsics + extrinsics
├── rosbag/
│   └── *.db3                             # exactly ONE; APS triangulations
├── labels/
│   └── *.pt                              # one or more torch dicts (see §2.1)
└── evs/
    ├── <cam_name>.raw                    # raw EVS stream per camera
    └── triggers.txt                      # produced by extract_triggers.py (§5)
```

Multiple recordings are typically grouped under a "recordings root":

```
<recordings_root>/
├── 20240510_101340_uechi_vs_li_set1/    ← one --root_dir
├── 20240510_103245_uechi_vs_li_set2/
└── ...
```

`extract_triggers.py` accept a
**recordings root** and recurse into every nested `evs/` or `label/`
folder; the labeling scripts (`label_pro_player_data*.py`) operate on
**one recording folder** at a time.

### 2.1 `labels/*.pt` schema

Each `.pt` file is a Python dict (load with
`torch.load(..., map_location="cpu", weights_only=False)`):

| Key                    | Shape / type           | Description                                          |
| ---------------------- | ---------------------- | ---------------------------------------------------- |
| `ball_position`        | `(N, 3)` float         | Raw APS-triangulated 3D ball position (world frame)  |
| `ball_orientation`     | `(N, 4)` float         | Quaternion of the ball                               |
| `ball_timestamps`      | `(N,)` float seconds   | APS frame timestamps (≈ 200 Hz, 5 ms cadence)        |
| `est_ball_position`    | `(N, 3)` float         | Smoothed/estimated ball position (post-processed)    |
| `events`               | list[dict]             | Contact/bounce events with `timestamp` keys          |
| `sequence_number`      | int                    | Rally / sequence id                                  |


### 2.2 Calibration YAML

A single `*.yaml` at the recording root that carries per-camera intrinsics and extrinsics. 
It is parsed by `tools/interpolate_ball_positions.py` during `project_3d_to_2d` to back-project the 1000 Hz 3D trajectory into the requested EVS camera frame.

### 2.3 ROS bag (`rosbag/*.db3`)

Required only for step 3 (events). The pipeline filters two topics:

- `/sensors/ball_triangulation/points`
- `/sensors/ball_pose_estimation/poses`

It uses **the timestamp of the first message** as the master `start_time`
that anchors APS-frame indices to EVS trigger indices (see §4).

### 2.4 EVS raw recordings

Standard Prophesee `.raw` files captured with external trigger input
wired to the APS camera shutter (see §4). One file per camera, named
`<cam_name>.raw`. The cameras this capture setup uses are
`evs00050026`, `evs00050027`, `evs00050028`, `evs00050034`.

---

## 3. Output data layout

After running steps 1–3 the recording folder gains:

```
<recording_folder>/
├── metadata/
│   └── <cam_name>/
│       └── seq_<NNN>_<label>.csv         ← per-rally CSV (step 1)
└── h5/
    └── <cam_name>/
        ├── seq_<NNN>_<label>.h5          ← events       (step 3)
        └── seq_<NNN>_<label>_label.h5    ← 2D labels    (step 1)
```

In addition `extract_triggers.py` writes `evs/triggers.txt` per recording.
`label_pro_player_data.py` also writes diagnostic plots under
`./<recording_name>/<cam_name>/polyfit_<deg>deg_stitch[...]/` in the
current working directory (or under `--plot_dir` when supplied).

### 3.1 Metadata CSV (`seq_<NNN>_<label>.csv`)

One row per 1000 Hz sample inside the rally. Columns:

| Column                   | Type   | Notes                                                    |
| ------------------------ | ------ | -------------------------------------------------------- |
| `sequence_id`            | int    | Same value on every row (rally id, zero-padded in name)  |
| `label_name`             | str    | Stem of the source `.pt` file                            |
| `timestamps`             | float  | Seconds, 1 ms cadence after polyfit upsampling           |
| `first_label_timestamp`  | float  | Rounded to 5 ms; rally start (APS-aligned)               |
| `last_label_timestamp`   | float  | Rounded to 5 ms; rally end                               |
| `pos_{x,y,z}_label`      | float  | 3D ball position (world frame)                           |
| `vel_{x,y,z}_label`      | float  | 3D ball velocity                                         |
| `orientation_{x,y,z}_label`      | float  | 3D ball velocity                                         |

`label_pro_player_data_events.py` asserts `sequence_id`, `label_name`,
`first_label_timestamp` and `last_label_timestamp` are **constant** in
each CSV (one CSV ↔ one rally).

### 3.2 Label H5 (`seq_<NNN>_<label>_label.h5`)

Written by `H5WriterLabel`
([tools/h5_writer.py](tools/h5_writer.py)). Datasets:

| Dataset       | Shape       | Description                                       |
| ------------- | ----------- | ------------------------------------------------- |
| `points`      | `(M, 2)`    | 2D back-projected ball center (px)                |
| `radius`      | `(M,)`      | 2D ball radius (px)                               |
| `velocities`  | `(M, 2)`    | 2D back-projected ball velocity                   |
| `timestamps`  | `(M,)`      | Seconds (synchronized with APS clock)             |
| `points_orig` | `(M, 2)`    | 2D projection of raw (un-interpolated) APS points |
| `radius_orig` | `(M,)`      | Same as above, original radius                    |

`M` is one entry per millisecond of rally duration (after polyfit upsample).

### 3.3 Event H5 (`seq_<NNN>_<label>.h5`)

Written by `H5Writer`
([tools/h5_writer.py](tools/h5_writer.py)). Datasets:

| Dataset       | Shape   | dtype | Description                                   |
| ------------- | ------- | ----- | --------------------------------------------- |
| `events/x`    | `(E,)`  | u2    | Pixel x                                       |
| `events/y`    | `(E,)`  | u2    | Pixel y                                       |
| `events/p`    | `(E,)`  | u1    | Polarity ∈ {0, 1}                             |
| `events/t`    | `(E,)`  | i8    | Microseconds, **relative to `t0`**            |
| `t0`          | `(1,)`  | i8    | Microseconds; `t[i] + t0 = absolute event ts` |
| `ms_to_idx`   | `(K,)`  | u4    | Cumulative index lookup, one per millisecond  |

`ms_to_idx[k]` is the first index `i` for which `t[i] / 1e3 >= k`,
i.e. the start of millisecond `k`. 

The `Events` dataclass enforcing these dtypes lives in
[tools/event_data_format.py](tools/event_data_format.py).

---

## 4. EVS ↔ APS time synchronization

The whole pipeline lives or dies on the **shared clock between APS frames
and EVS events**. The mechanism is:

1. **Hardware-level trigger.** The APS camera's frame-start signal is
   wired into the EVS camera's external-trigger input. Every APS shutter
   produces an "external trigger" event in the EVS stream (rising edge,
   `p == 1`).
2. **`triggers.txt` extraction.**
   [src/evs/tools/evs_trigger_to_txt](../evs/tools/evs_trigger_to_txt) reads
   the `.raw` file, keeps trigger events with `p == 1`, drops duplicates
   closer than `--min_dt_us` (default 1000 µs), and writes the trigger
   timestamps (in EVS clock microseconds) one per line.
   It also adds `exposure_time_us / 2` (default 750/2 = 375 µs) so the
   stamp lines up with the **midpoint of the APS exposure** rather than
   with the shutter rising edge.
3. **APS clock anchor.**
   `tools/extract_aps_triangulations.py::extract_first_aps_frameid` reads
   the rosbag and grabs the timestamp of the first ball-triangulation
   message — call this `start_time` (seconds, ROS clock).
4. **Index correspondence.** APS runs at 200 Hz (5 ms / frame), so
   APS frame `i` of a recording corresponds to trigger index

   ```python
   gt_frameid = round((aps_ts - start_time) * 1e3 / 5)
   ```

   where `gt_frameid` is the **index into `triggers.txt`** at which the
   matching event window starts. `label_pro_player_data_events.py`
   asserts that the `gt_frameid` list is contiguous (`np.diff == 1`)
   for every rally.
5. **Window iteration.**
   [`FixedSizeTriggerEventReader`](../evs/evs/utilities/event_readers.py)
   iterates the `.raw` file one trigger-bounded window at a time. For
   each window whose index sits in `gt_frameid` for the current rally,
   the events script copies the events into the output H5.
6. **Sub-frame trimming.** APS labels can start/end **inside** a trigger
   window. The script computes
   - `first_time_diff = first_aps_ts - first_label_ts` (≤ 5 ms)
   - `last_time_diff  = last_label_ts  - last_aps_ts`  (≤ 5 ms)
   and trims the **first/last** event windows to the exact label
   boundaries. The first kept event timestamp is stored as `t0` in the
   output H5; all subsequent timestamps are written as `t - t0` in µs.

### 4.1 Why some recordings drift

Tokyo recordings (older dataset) had inconsistent APS frame intervals
that did **not** match the EVS triggers exactly, producing a constant
shift between event and label across a rally. 

---

## 5. Pipeline stages in detail

### 5.1 Step 1 — `tools/extract_triggers.py`

Input: any directory tree that contains `evs/` folders with `*.raw`.
Output: one `triggers.txt` per `evs/` folder.

```bash
python src/data_generation/tools/extract_triggers.py \
    --root_dir <recordings_root_or_one_recording> \
    --trigger_tool /path/to/workspace/src/evs/tools/evs_trigger_to_txt
```

The script recursively finds every `evs/` folder and runs
`evs_trigger_to_txt` against each `.raw` inside, appending all triggers
into a single `evs/triggers.txt`. Folders that already contain a
`triggers.txt` are skipped (toggle by editing the early-return).

The `--trigger_tool` default points at the source-tree script, **not**
the install/share copy, so you don't need to rebuild the workspace.

### 5.2 Step 2 — `label_pro_player_data.py`

Input: `labels/*.pt`, `*.yaml` calibration.
Output: `metadata/<cam>/*.csv`, `h5/<cam>/*_label.h5`, plots.

For each `.pt`:

1. Discard if smoothed vs. raw RMS divergence > 1 cm.
2. Split the `(N, 3)` trajectory at NaN rows → contiguous non-NaN runs.
3. Within each run, split again at every contact/bounce event whose
   timestamp falls inside the run.
4. For each segment with ≥ `max(deg + 1, --min_samples)` samples:
   - Polynomial-fit (`--polyfit_degree`, default 3) per axis to
     interpolate to **1 ms** (= 1000 Hz / 500 Hz depending on flag),
     produce velocity by analytic derivative, plus a 200 Hz finite-diff
     velocity for sanity.
   - Stitch segments back into one continuous polyline.
5. Project the stitched 3D polyline into the camera frame via the
   provided YAML calibration → 2D `points`, `radius`, `velocities`.
6. Skip rallies with fewer than 11 in-view samples.
7. Write the label H5 + the per-rally CSV. `seq_id_counter` increases
   monotonically across rallies and is encoded in the filename and as
   the `sequence_id` column.

CLI flags worth knowing:

| Flag                | Default | Purpose                                            |
| ------------------- | ------- | -------------------------------------------------- |
| `--root_dir`        | —       | One recording folder (see §2)                      |
| `--cam_name`        | —       | Which EVS camera to back-project for               |
| `--ball_radius`     | 0.02 m  | Used to compute 2D radius                          |
| `--polyfit_degree`  | 3       | Per-segment polynomial degree                      |
| `--min_samples`     | 10      | Minimum APS samples per segment                    |
| `--plot_dir`        | cwd     | (Used by batch script) where to dump diagnostics   |

### 5.3 Step 3 — `label_pro_player_data_events.py`

Input: everything produced by steps 1 & 2 plus the rosbag and
`evs/<cam>.raw`.
Output: `h5/<cam>/seq_<NNN>_<label>.h5` (events).

Logic, per metadata CSV:

1. Read `start_time` (first APS triangulation message → also writes
   `start_time.txt` next to the rosbag).
2. Compute `gt_frameid` list (§4).
3. If the label window extends past the last APS frame, append one
   extra trigger window.
4. Iterate `FixedSizeTriggerEventReader(raw, triggers.txt)`:
   - Window `idx == gt_frameid[0]`: keep only the trailing slice
     `event_ts ≥ end - 1 ms - first_time_diff`, store `t0`.
   - Middle windows: keep all events.
   - Window `idx == gt_frameid[-1]`: keep only `event_ts ≤ start +
     last_time_diff` (or all if no fractional tail), then `break`.
5. Write events to `seq_<NNN>_<label>.h5` via `H5Writer`. `t0` lands in
   the file's metadata.

> **Heads-up:** `source_path_event_triggers` is currently commented out
> at the top of the script. Restore the line
> `source_path_event_triggers = Path(args.root_dir) / "evs" / "triggers.txt"`
> before running, otherwise you will hit `NameError`.

### 5.4 Visualization — `convert_h5_to_video_zoomed.py`

Renders an MP4 per `seq_*.h5` overlaying:

- Event time-surface (accumulation window = `--accumulation_time` ms).
- Polynomial-interpolated 2D ball position, radius, velocity arrow.
- (`--debug`) Original APS-triangulated 2D ball position in green.

`--label_offset` shifts label index vs. event window to compensate for
clock drift on bad recordings.


---

## 6. Batch execution

[run_pro_player_label_scripts_all.sh](run_pro_player_label_scripts_all.sh)
loops over every subfolder of a recordings root and every camera in
`CAM_NAMES`, runs steps 1 → 3 in sequence, and aborts the events step
if step 1 failed for that combination.

Edit the variables at the top of the script:

```bash
SETS_DIR="/path/to/<recordings_root>"
CAM_NAMES=("evs00050028" "evs00050027" "evs00050026" "evs00050034")
PLOT_DIR="/path/to/analysis_plots"
```

The matching single-recording version is
[run_pro_player_label_scripts.sh](run_pro_player_label_scripts.sh).

For visualization across many cameras, see
[run_pro_player_visualization.sh](run_pro_player_visualization.sh).

---

## 7. Conventions & invariants

- **APS cadence** is 5 ms (200 Hz). Rally label sampling after polyfit is
  1 ms (1000 Hz); event timestamps are µs.
- **Sequence id** (`seq_NNN`) is unique per recording / camera pair,
  three-digit zero-padded, increasing monotonically as `.pt` files are
  iterated alphabetically.
- **`t0` semantics.** Inside an event H5, the absolute event timestamp
  of sample `i` is `t0 + events/t[i]` (µs, EVS clock).
- **Calibration.** Exactly one `*.yaml` in the recording root.
- **Rosbag.** Exactly one `*.db3` under `rosbag/`.
- **Trigger polarity.** Only `p == 1` triggers are kept.
- **Reproducibility.** All `H5Writer*` constructors `assert` that the
  output path does not already exist; delete or move old files before
  re-running.

---

## 8. Common failure modes

| Symptom                                                                 | Fix                                                                                                  |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `IndexError: boolean index did not match indexed array, dim 0 is 0`     | `.raw` had no `p == 1` triggers. Patched in `evs_trigger_to_txt` (skip mask when `time.size <= 1`).  |
| `KeyError: 'sequence_id'`                                               | Old metadata CSV. Re-run `label_pro_player_data.py` after the `sequence_id` column patch.            |
| `NameError: source_path_event_triggers`                                 | Uncomment the assignment in `label_pro_player_data_events.py` (§5.3).                                |
| `AssertionError: not Path(outfile).exists()`                            | Output H5 already exists; remove it or write to a fresh `--root_dir`.                                |
| `assert (np.diff(gt_frameid) == 1).all()` fails                         | Missing/duplicate APS frames in the rally. Inspect the `.pt` `ball_timestamps`; consider re-labeling.|
| `assert last_label_ts > last_aps_ts`                                    | Rounding mismatch around rally end. Check 5 ms vs. 1 ms rounding alignment.                          |
| `est_mean > 0.01` (rally silently skipped)                              | Smoothed vs. raw 3D ball position diverges ≥ 1 cm; expected for noisy rallies.                       |

---

## 9. File reference (this folder)

| Script / module                                                                     | Stage      | Purpose                                                  |
| ----------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------- |
| [label_pro_player_data.py](label_pro_player_data.py)                                | 1          | Polyfit, project, write metadata CSV + label H5          |
| [tools/extract_triggers.py](tools/extract_triggers.py)                              | 2          | Recursively run `evs_trigger_to_txt` per `evs/` folder   |
| [tools/extract_aps_triangulations.py](tools/extract_aps_triangulations.py)          | 3 (helper) | Read first APS message timestamp from the rosbag         |
| [label_pro_player_data_events.py](label_pro_player_data_events.py)                  | 3          | Slice raw events into rally H5s using triggers + CSVs    |
| [tools/h5_writer.py](tools/h5_writer.py)                                            | 1, 3       | `H5Writer` (events) and `H5WriterLabel`                  |
| [tools/event_data_format.py](tools/event_data_format.py)                            | 3          | `Events` dataclass with dtype contracts                  |
| [tools/interpolate_ball_positions.py](tools/interpolate_ball_positions.py)          | 1          | Polyfit, finite-diff, 3D→2D projection                   |
| [convert_h5_to_video_zoomed.py](convert_h5_to_video_zoomed.py)                      | viz        | Per-sequence MP4 with event time-surface + labels        |
