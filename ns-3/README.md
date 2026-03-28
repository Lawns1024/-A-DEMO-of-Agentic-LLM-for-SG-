## Experiment Overview

This folder contains the ns-3 experiment assets for the CEVI IoT MAC evaluation.

## How to Run

1. Copy `cevi-iot-mac.cc` into the `scratch/` directory of your ns-3 workspace.
2. Run the ns-3 simulation and make sure it produces `cevi-iot-policy.csv`.
3. Generate plots using:
   - `python plot_cevi_iot.py --csv /path/to/cevi-iot-policy.csv`

## Outputs

- `cevi-iot-metrics.png`
- `cevi-iot-metrics.pdf`

## CSV Column Definitions

- `time_s`: Simulation time (seconds)
- `node_id`: Node ID
- `state`: Policy state
- `cw_min` / `cw_max`: Contention window bounds
- `throughput_mbps`: Throughput (Mbps)
- `collision_rate`: Collision rate (0–1)
