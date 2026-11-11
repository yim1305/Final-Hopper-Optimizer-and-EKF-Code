# Guidance & Navigation — Hopper1

The G&N (guidance and navigation) simulation reads the validated **8-A2200, 2,000 kg dry DEM flight** produced by Astro (`Astro/results/a2200_08/`). It simulates the vehicle's navigation sensors with their errors: the LN-200S inertial measurement unit (IMU), two star trackers and terrain-relative navigation (TRN) position fixes supplied by SPLICE. A 15-state error-state extended Kalman filter (EKF) combines them, and a Monte Carlo study (many repeated runs with random sensor errors) shows how well the vehicle knows its position, velocity and attitude along the flight, especially at touchdown.

Navigation estimates do not feed back into Astro's optimized flight: GN evaluates navigation knowledge on a fixed reference trajectory. It is not a closed-loop guidance or control simulation.

Contents: [Folder and file map](#folder-and-file-map) · [Running it](#running-it) · [Input and truth](#input-and-truth) · [Study settings](#study-settings) · [Scope and interpretation](#scope-and-interpretation) · [Tests](#tests) · [Recorded results and history](#recorded-results-and-history)

## Folder and file map

All paths are relative to `GN/`.

| Path | Role |
|---|---|
| `nav_config.yaml` | All GN settings: which Astro case to read, the output folder, sensor and error values, the six sensor cases, the Monte Carlo seed and run count, the TRN sweep grid and the requirement. Unknown values are marked `PLACEHOLDER`. |
| `gn_sim/` | The simulation package (below). |
| `scripts/` | Launchers to run from the repository root. They add the `GN` and `Astro` source folders to the import path, so no manual `PYTHONPATH` is needed. |
| `tests/` | Acceptance tests (see [Tests](#tests)). |
| `results/` | Generated locally by the scripts (default `results/dem_a2200_08/`). Not stored. |

### `gn_sim/` package

| File | Purpose |
|---|---|
| `config.py` | Loads `nav_config.yaml` and converts the human units to SI. |
| `truth.py` | Loads and verifies the accepted Astro flight, builds the 6-DOF kinematic truth hop by hop (knot-by-knot propagation, prescribed and smoothed attitude, checkpoints). |
| `sensors.py` | Measurement models: IMU increments, star-tracker measurements and availability, SPLICE TRN position fixes, IMU range and update-gap checks. |
| `ekf.py` | Strapdown propagation and the 15-state multiplicative error-state EKF with Joseph-form measurement updates, vectorized over Monte Carlo runs. |
| `montecarlo.py` | Runs one sensor case on one hop for N Monte Carlo runs and summarizes the errors. |
| `rotations.py` | Batched rotation helpers (exponential/logarithm maps, skew matrices). |
| `plots.py` | Static PNG figures: error histories, case comparison, star-tracker value, touchdown distribution, tracker timeline, TRN sweep heat map. |
| `scripts/run_nav.py` | Main study: all six cases on both hops. |
| `scripts/run_trn_sweep.py` | Sweep of TRN accuracy and update rate. |
| `scripts/run_design_checks.py` | One-at-a-time sensitivity studies on case C. |
| `__init__.py` | Marks the package and points the matplotlib cache inside the repository (`.maintenance/`, which `.gitignore` excludes and which is created only when a script runs). |

## Running it

**All commands are run from the repository root** with `.\.venv\Scripts\python.exe -B ...`. The launchers in `GN/scripts/` add the `GN` and `Astro` source folders to the import path, so no `PYTHONPATH` setting is needed. Do not run the `gn_sim/scripts/*.py` files directly; they are the code that the launchers call.

**Prerequisite for every GN command: the generated Astro flight.** GN cannot run from a fresh clone alone. A fresh clone holds only the restart arrays `Astro/results/a2200_08/sizing.npz` and `duals.npz`. Run Astro's validation of `a2200_08` first:

```powershell
.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py a2200_08
```

This writes `Astro/results/a2200_08/summary.json`, `trajectory.npz`, `parameters.json`, `validation.json` and `trackers.json`, which GN reads (the stored `sizing.npz` is read as well). Do not substitute a different optimizer run for this flight.

**Order and details.** All outputs go to the folder named by `outputs.directory` in `nav_config.yaml`, which is `GN/results/dem_a2200_08/` (generated locally, not stored). The input case is `trajectory.case_dir` in the same file (`Astro/results/a2200_08`).

| Order | Command | Needs first | Writes (in `GN/results/dem_a2200_08/`) | Runs from a fresh clone? |
|---|---|---|---|---|
| 0 | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py a2200_08` | The stored restart arrays | See [Astro/README.md](../Astro/README.md) | Yes |
| 1 | `.\.venv\Scripts\python.exe -B GN/scripts/run_nav.py [--runs N]` | Step 0 | `summary.json`, `RESULTS.md`, `histories.npz`, `touchdown_runs.npz`, and the figures `comparison_ABC.png`, `errors_<case>_hop<n>.png`, `star_tracker_hop<n>.png`, `star_tracker_value.png`, `touchdown_distribution.png` | No, only after step 0 |
| 2 | `.\.venv\Scripts\python.exe -B GN/scripts/run_trn_sweep.py [--runs N]` | Step 0 (independent of step 1) | `trn_sweep.json`, `TRN_SWEEP.md`, `trn_sweep.png` | No, only after step 0 |
| 3 | `.\.venv\Scripts\python.exe -B GN/scripts/run_design_checks.py [--runs N]` | Step 0, and the output folder must already exist (run step 1 or step 2 first) | `design_checks.json`, `DESIGN_CHECKS.md` | No, only after steps 0 and 1 or 2 |
| 4 | `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider GN/tests` | Step 0 | Nothing | No, only after step 0 |

`run_nav.py` and `run_trn_sweep.py` create the output folder; `run_design_checks.py` does not, and would stop with a file-not-found error after finishing its computation if the folder is missing. All three launchers accept `--runs N` to override the Monte Carlo run count. Without it, `run_nav.py` uses the 100 runs per case and hop from `nav_config.yaml`, `run_trn_sweep.py` uses its configured 50 runs per grid cell, and `run_design_checks.py` uses 50 runs per value. The studies never write into `Astro/`.

Each study saves its effective settings and results in `summary.json`, with numerical histories and touchdown arrays, checkpoint and consistency tables, and error and availability plots.

## Input and truth

- `nav_config.yaml` (`trajectory:` section) selects the Astro case (`Astro/results/a2200_08`, branch `uprange`), the GN output folder and the existing sensor and error assumptions.
- The loader reads the complete `trajectory.npz`, the effective `parameters.json`, `summary.json`, `validation.json` and `trackers.json`. It verifies the physical, source and data identities (SHA-256 hashes) and requires successful flight, reserve and tracker acceptance before using the flight. If any hashed Astro file was edited since the flight was saved, GN refuses to load it until `run_case.py a2200_08` is re-run.
- GN imports Astro's `hopper/` and `mission/` helpers read-only. Astro never imports GN.
- Translation uses the same force model and optimized linear controls as Astro, integrated independently at every control knot. Terminal propagation stops at first descending DEM contact. Altitude, descent checkpoints and TRN altitude gating use the actual height above the DEM, including locations below the reference lunar radius.
- Attitude is prescribed from Astro's thrust-aligned frame and coast interpolation. Phase-join discontinuities are smoothed at the vehicle slew setting, 6 deg/s (Astro `limits.tilt_rate_max_deg_s`). This is kinematic truth, not torque dynamics or a controller.
- Star-tracker Sun/Earth exclusions and full-cone terrain obstruction use the same local native DEM and global LOLA geometry as the accepted flight. These are sampled cone/ray checks. GN uses the configured **6 deg/s hardware limit**, the same value Astro enforces.

The navigation filter is initialized independently at each launch. Position and velocity start nominally known, with configured 1 mm and 1 mm/s numerical uncertainties; initial attitude uncertainty comes from tracker head 1 in every case, including E. Hop 1's landing error is not carried into hop 2. The cumulative inertial epoch and zero surface stay retain Astro's conventions. Landing ground is assumed flat at DEM-derived elevation with radial local up. No landing-gear or contact-load model and no GNC tracking simulation is added.

Truth and filter use the Moon-centered inertial frame I, aligned with the Moon-fixed frame F at first launch; lunar rotation relates them by `r_I = Rz(omega_moon * t) r_F`. Checkpoint errors are resolved locally into downrange, crossrange and vertical components. Truth and filter share central gravity; gravity anomalies are omitted.

## Study settings

IMU increments at 400 Hz are accumulated to the 16 Hz filter step. Two tracker heads provide attitude measurements at their configured 16 Hz output rate; the nominal filter uses one IMU, with the second a spare. The six sensor cases of `nav_config.yaml` are:

| Case | Sensors |
|---|---|
| A | IMU only (dead reckoning) |
| B | IMU + two star trackers |
| C | IMU + two star trackers + SPLICE TRN |
| D1 | IMU + head 1 only + SPLICE TRN |
| D2 | IMU + head 2 only + SPLICE TRN |
| E | IMU + SPLICE TRN, no star trackers |

Sensor noise and bias values, initial uncertainties, Monte Carlo seeds and the TRN accuracy/rate grid are set in `nav_config.yaml`. IMU errors are a constant turn-on bias per run plus white noise; TRN fixes are isotropic white-noise position measurements with zero latency. Supplied sensor specifications are treated as 1-sigma because their sigma level was unspecified. Unspecified inputs remain labeled placeholders in `nav_config.yaml`: the default TRN accuracy of 50 m per-axis 1-sigma at 0.2 Hz, and the IMU velocity random walk (VRW) of 0.03 m/s/sqrt(hr). SPLICE supplies the TRN position data; the hopper's separate hazard-camera viewing constraints are not a model of TRN image processing. TRN is configured as available throughout the flight; the team's high-altitude capability check still needs its supporting source when presenting.

The seeded Monte Carlo varies sensor errors on one fixed flight, not thrust or trajectory dispersions. Single-head cases disable the other head for the entire hop. IMU spare switching is not simulated. The optional sweep uses TRN sigma {10, 30, 100, 300} m and rate {0.05, 0.1, 0.2, 1} Hz, with 50 runs per cell.

## Scope and interpretation

The 50 m requirement used throughout is horizontal navigation knowledge at touchdown. It is separate from Astro's 25 m nominal landing tolerance. The saved Monte Carlo horizontal "3-sigma" metric is three times the horizontal RMS error, not an empirical 99.7th percentile. Results do not establish a combined landing-error budget or closed-loop landing performance. Monte Carlo spread and filter covariance are separate evidence; neither proves performance for unmodeled errors.

## Tests

`GN/tests/test_gn_sim.py` holds the acceptance checks. They cover:

- the saved input being the validated 2,000 kg dry `a2200_08` flight, and TRN using the DEM height below the reference sphere;
- the rotation exponential/logarithm round trip;
- a perfect IMU reproducing the truth trajectory and attitude, and the truth running from launch to touchdown;
- a hand check that a constant accelerometer bias gives a dead-reckoning position error of 0.5 b t^2;
- star-tracker availability (a higher rate gate cannot reduce availability) and the batched terrain-cone check matching Astro's fine ray test;
- filter consistency for case C (per-axis fraction within 3-sigma above 0.98, mean NEES between 2 and 4; NEES is the normalized estimation error squared).

The tests need the generated Astro files from the prerequisite step above.

## Recorded results and history

These numbers were produced locally on earlier runs; the result folders themselves are not stored. All use the touchdown horizontal Monte Carlo metric (3 x horizontal RMS) and the filter 3-sigma, reported as outbound / return, against the 50 m comparison, with the placeholder TRN inputs above.

**Current flight (5,363.13 kg, 2026-11-04 and 2026-11-05 runs):**

- Main study (100 runs per case and hop, sensors unchanged): case C Monte Carlo 39.5 / 44.0 m (vertical 27.5 / 29.7 m), filter 40.3 / 42.0 m; no run over 50 m (worst 23.4 / 34.2 m). Case E (no trackers) 81.3 / 89.3 m fails. Cases A and B (dead reckoning) drift 2.1 to 2.8 km. GN tests passed 12 of 12 on this flight (the Astro hashes matched, so the loader accepted the flight).
- Tracker gaps are longer than on the previous flight: whole-flight maximum 2.06 s on hop 1 (t = 651 s) and 4.75 s on hop 2 (t = 594 s, in the final descent), against the 5 s update-age placeholder. The hop-2 gap sits 0.25 s under the limit, so the 5 s limit is now a binding assumption.
- TRN sweep (50 runs per cell, worse of Monte Carlo and filter 3-sigma, worse hop): passes for <= 10 m at any rate, <= 30 m at >= 0.1 Hz (0.05 Hz now fails, 53 m), <= 100 m only at 1 Hz (45 m).
- Design checks (case C, 50 runs): nominal 49.1 m on the worst hop (hop 2, Monte Carlo), so the margin is about 1 m at 50 runs. A TRN altitude ceiling <= 10 km fails (56 to 85 m); launch-position 1-sigma >= 10 m fails (52.5 to 53.9 m); VRW 0.1 fails (51.9 m) and 0.3 fails (62.0 m).
- The TRN sweep and design checks were rerun on 2026-11-05 on this flight with the same seed; the main study was reproduced exactly (case C Monte Carlo 39.5 / 44.0 m). Until then the sweep and design checks still described the earlier flight.
- Per-run touchdown statistics, case C (100 runs): horizontal median 12.0 / 13.0 m, 95th percentile 20.5 / 25.7 m, worst 23.4 / 34.2 m. The 3-sigma (3 x RMS) metric and an empirical 95th-percentile metric give different pass margins, so any report must state which metric it uses.
- Assumptions of the descent variant that GN inherits: 2224 N below the 4448 N floor means two engines off and two at 50 % (same propellant as four at 25 %); throttle slew is kept at 1483 N/s although two engines allow 741 N/s (exceeded for 0.7 s per observation phase, 5.5 s in the hop-1 terminal phase and 7.4 s in the hop-2 terminal phase); engine shutdown and relight are not modeled; the 5 deg/s pointing margin is solver-side only (the camera check uses 6 deg/s); touchdown is at the 1.5 / 0.5 m/s limits with thrust equal to weight. See `Astro/README.md`.

**Earlier flight (5,309.64 kg, 6 deg/s, 2026-10-28; superseded by the flight above).** Touchdown horizontal Monte Carlo metric, 100 runs per case and hop, outbound / return:

| Case | Sensors | Outbound | Return |
|---|---|---:|---:|
| A | IMU only | 2,690.7 m | 2,090.6 m |
| B | IMU + two trackers | 2,647.4 m | 2,142.2 m |
| C | IMU + two trackers + TRN | 37.7 m | 40.5 m |
| D1 | IMU + head 1 + TRN | 40.3 m | 40.9 m |
| D2 | IMU + head 2 + TRN | 37.8 m | 42.4 m |
| E | IMU + TRN, no tracker updates | 78.7 m | 82.5 m |

Case C filter horizontal 3-sigma was 40.2 / 42.0 m, and zero case C runs exceeded 50 m out of 100 on each hop (a finite sample, not a reliability guarantee). The TRN sweep on that flight gave the requirement <= 30 m 1-sigma at >= 0.05 Hz or <= 100 m at 1 Hz. One-at-a-time case C sensitivities (50 runs per value): TRN must provide fixes during the roughly 17 km coast (any altitude ceiling <= 10 km fails, 58 to 84 m); launch-position 1-sigma up to 100 m changes touchdown knowledge by about 3 m; accelerometer VRW up to 0.1 m/s/sqrt(hr) passes and 0.3 fails. Whole-flight either-head tracker gaps were 1.50 / 1.25 s and ascent-plus-coast gaps 1.08 / 0.93 s, so no gap exceeded 2 s; GN's 16 Hz schedule finds at most 1.5 s, while Astro's coarser 1 s candidate grid finds 3.0 s because the early pitch-over rides the rate limit. Phase-join attitude steps were 5.9 / 3.6 deg, slewed at 6 deg/s. An earlier study on 2026-10-27 (spherical-model predecessor of the DEM input) gave case C 40.33 / 43.72 m Monte Carlo and 40.18 / 42.29 m filter.

**Other recorded points:**

- Case C per-axis coverage is about 99.7 to 99.8 % within filter +/-3-sigma, with mean three-component NEES about 3.0 to 3.1 (6 deg/s flight).
- Why the trackers help: attitude updates limit gyro-driven thrust-direction error, but they do not measure position; TRN provides the position aiding. For a constant accelerometer bias b, position drift grows approximately as 0.5 b t^2; the implemented filter gives the actual results.
- Sensor sources still to be confirmed: the LN-200S errors come from a listing of the older LN-200 (its applicability to the LN-200S needs checking), and the high-altitude availability of SPLICE TRN needs the team's supporting source. Hardware figures retained from earlier team notes, pending source verification: two IMUs at 0.748 kg and 12 W each and two trackers at 1.8 kg and 6 W each, totaling 5.096 kg and 36 W with all four powered. SPLICE hardware is counted separately. These are not newly verified specifications and are not added to Astro's mass.
- Star-tracker mounting review (2026-11-08): the saved `a2200_08` mounts are tilted 17.3399 / 35.0643 deg from body +Z (elevations 72.6601 / 54.9357 deg above the body XY plane), at azimuths 0 / 180 deg, giving a 52.4042 deg angle between the boresights. Two offset heads have flight precedent: PROBA used a 50 deg separation (The PROBA satellite star tracker performance, https://www.sciencedirect.com/science/article/pii/S0094576504002942). Exact tilts are mission-specific. Still to be verified: physical baffle and structure clearance, thermal alignment stability and lunar stray-light performance; the boresight geometry alone does not qualify the mechanical installation.
- Mount-angle provenance: `mission/trackers.py` reads the configured angles and validates them on the saved DEM flight, which is a feasibility check, not a mount search. Earlier simulations are reported to have optimized the mounting angles as free variables, and historical four-angle search code remains in `hopper/star_tracker.py`, but the original search report that would establish 72.6601 / 54.9357 deg as an optimum was not located. The later Astro mounting-reconstruction study (described in `Astro/README.md`) is a separate post-processing availability study on the unchanged flight; no sensor settings, flight results or GN results changed.
