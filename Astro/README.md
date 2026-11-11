# Astro — Hopper1 trajectory optimizer

Astro minimizes the wet mass (dry mass plus loaded propellant) of the Hopper1 lunar hopper and solves the reference position, velocity, mass and thrust history for an outbound hop and the return hop. The terrain comes from a 30 m digital elevation model (DEM) of the lunar south pole region. Landing ground is assumed flat at its DEM-derived elevation. Astro is a trajectory optimization, not a guidance, navigation and control (GNC) simulation.

For every accepted case, Astro also validates the optimized flight independently: it re-integrates the flight with a separate integrator, checks the touchdown limits, the hazard camera view, the engine hardware envelope, the propellant reserve and the star-tracker update schedule.

The selected design is the 8-A2200 case (eight A2200 engines, 2,000 kg dry mass), called `a2200_08`. Its trajectory figures are shown in the [repository README](../README.md).

Contents: [Folder and file map](#folder-and-file-map) · [Running each case](#running-each-case) · [Model and fixed assumptions](#model-and-fixed-assumptions) · [Optimization and acceptance](#optimization-and-acceptance) · [Descent-thrust variant](#descent-thrust-variant-opt-in-diagnostic) · [Status of each case](#status-of-each-case) · [Inactive joint-solver experiment](#inactive-joint-solver-experiment) · [Recorded results and history](#recorded-results-and-history)

## Folder and file map

All paths are relative to `Astro/`.

| Path | Role |
|---|---|
| `mission.yaml` | The active common mission, the five architectures (cases) and the analysis settings. Also holds the `descent_variant` and `tracker_solver` sections. |
| `config.yaml` | Baseline parameter values read by the frozen core. The active mission and architecture values in `mission.yaml` override it. |
| `hopper/` | Frozen optimization core (see below). Read-only: never edit it. |
| `mission/` | The mission workflow built on top of `hopper/` (see below). |
| `scripts/` | Command-line entry points and study scripts (see below). |
| `tests/` | Regression tests (see below). |
| `data/` | Lunar terrain inputs (see below). |
| `results/` | Only the restart arrays of the two stored cases (`a2200_08`, `a2200_12`). Everything else that this README says is written under `results/` is generated locally by the named script and is not stored. |

### `hopper/` — frozen core

The core defines the dynamics, constraints, engine model and the Hermite–Simpson collocation (a method that turns the continuous optimal control problem into a nonlinear program), solved with IPOPT through CasADi. Saved flights hash every file in this folder, and the trajectory cache identity is computed from its source (`hopper/tracker_cache.py`), so any change to it invalidates saved results. Do not add or edit files here.

| File | Purpose |
|---|---|
| `params.py` | Loads `config.yaml` into frozen dataclasses and derives values. |
| `dynamics.py` | 3-DOF dynamics, written once and used for both CasADi (collocation) and numpy/scipy (validation). |
| `problem.py` | The round-trip optimal control problem: hop 1 (start site to a gate over the target to the divert site), then hop 2. |
| `phases.py` | Phase container: the decision variables of one trajectory segment and its constraint list. |
| `constraints.py` | The constraints, each as a class with `apply(opti, phase)`. |
| `transcription.py` | Direct transcription of the dynamics into collocation defects. |
| `staging.py` | Relax, stage, refine procedure for fixed-thrust (non-pulsing) engines. |
| `initial_guess.py` | Initial guess: Lambert coast arc plus rocket-equation burn estimates, with cubic-Hermite state interpolation. |
| `lambert.py` | Universal-variable Lambert solver (short way, single revolution) and the impulsive hop baseline. |
| `divert.py` | Cost of the divert descent from a fixed gate state to a landing site. |
| `sensor.py` | Square-field-of-view hazard camera geometry; thrust is the body-up proxy and roll follows a fixed heading. |
| `star_tracker.py` | Two fixed tracker directions, frames, mounting search and update schedules. |
| `tracker_params.py` | Optional numerical mounting-study settings. |
| `star_tracker_coupled.py` | Local joint flight/mount nonlinear program with fixed labeled update cells (used only by the inactive joint-solver experiment). |
| `tracker_cache.py` | Versioned numeric-only trajectory cache and the source-identity hash. |
| `postprocess.py` | Delta-v, propellant and burn-timeline bookkeeping from a solved round trip. |
| `validate.py` | Independent check: re-propagates the optimized controls with scipy and compares with the collocation. |
| `plotting.py` | Matplotlib views of a solved round trip. |
| `environment/gravity.py`, `environment/terrain.py` | Gravity models, and a terrain model with an altitude helper. |

### `mission/` — mission workflow

These files are also hashed into saved flights.

| File | Purpose |
|---|---|
| `config.py` | Builds the effective inputs of each case from `mission.yaml` and `config.yaml`; also file hashing and result-writing helpers. |
| `terrain.py` | Native DEM interpolation and the terrain constraints of the accepted flight (`GroundProblem`). |
| `run.py` | The sizing and validation workflow for all cases; this is what `scripts/run_case.py` runs. |
| `validation.py` | Independent integration of each control interval, first-contact detection with the DEM, and the camera check. |
| `hardware.py` | Configured engine envelopes and operating-life checks. |
| `trackers.py` | Terrain-aware star-tracker schedules on the solved flight. |
| `report.py` | Short flight reports and standard plots from accepted trajectories. |
| `los.py` | Terrain-only radio visibility from the flight to the launch base. |
| `tracker_solver.py` | Tracker events coupled to the trajectory optimization. Used only by the inactive joint-solver experiment. |
| `__init__.py` | Marks the package and points the matplotlib cache inside the repository (`.maintenance/`, which `.gitignore` excludes and which is created only when a script runs). |

### `scripts/` — entry points and studies

The exact command, prerequisites and output of every script are in [Running each case](#running-each-case). Output folders named in this section are created locally when the script runs.

Main entry points:

| Script | What it does |
|---|---|
| `run_case.py` | **Main entry point.** Runs, validates or re-reports one case (`a2200_08`, `a2200_10`, `a2200_12`, `a2200_14`, `ar46_leros_rcs`, or `all`). Options: `--report-only`, `--retry`. |
| `run_los.py` | **Main entry point.** Terrain-only radio line of sight from the accepted `a2200_08` flight to the launch base. Writes `results/los/`. |

Studies on the accepted flight (supporting scripts, each writing only its own folder under `results/`, except `run_dv_losses.py`, which writes one file into the case folder):

| Script | What it does |
|---|---|
| `run_hop_distance.py` | Hop-distance demonstration: flies the 8-A2200 design 50, 100 and 150 km from the same launch base toward Drygalski and applies the same acceptance checks. Takes optional case names `d050 d100 d150` and `--report-only`. Writes `results/hop_distance/`. It starts from a built-in Lambert-based guess, so it does not need restart arrays. |
| `run_design_checks.py` | Design-verification variants of `a2200_08`: dry mass -10 % and +10 % (`dry1800`, `dry2200`) and a mesh doubled in every phase except ascent (`mesh_other2x`). Takes optional variant names and `--report-only`. Seeded from `results/a2200_08`. Writes `results/design_checks/`. |
| `run_tracker_sun_sweep.py` | Star-tracker availability on the `a2200_08` flight over a sweep of Sun and Earth directions. A post-check: the trajectory and mounts are unchanged. Writes `results/tracker_sun_sweep/a2200_08/`. |
| `run_earth_los.py` | Direct-to-Earth line of sight along the saved flight and at both landed sites over the libration range, using terrain only (no link budget). Writes `results/earth_los/`. |
| `run_dv_losses.py` | Splits each hop's finite-burn delta-v excess over the Lambert minimum into gravity loss, steering loss and a remainder. Writes `results/<case>/dv_losses.json`. |
| `reconstruct_tracker_mounts.py` | Evaluates alternative star-tracker mounts on the saved `a2200_08` flight without changing the trajectory. Options include `--case`, `--out`, `--report-only`, `--resume-grid`. Default output `results/mount_reconstruction/a2200_08/`. |

Descent-thrust variant tools (see [Descent-thrust variant](#descent-thrust-variant-opt-in-diagnostic)):

| Script | What it does |
|---|---|
| `descent_ladder.py` | Walks one case from a saved solution to the `mission.yaml` descent settings by continuation, then runs the acceptance check. Writes only `results/ladder_<case>/`. |
| `run_descent_variant.py` | Solves one diagnostic run with a lower descent thrust floor and a touchdown thrust-to-weight limit. No acceptance checks. Writes `results/<tag>/`. |
| `check_descent_variant.py` | Runs the full `run_case` acceptance checks on a saved descent-variant run. Writes `results/<tag>/acceptance.json`. |

Supporting scripts used during earlier case work:

| Script | What it does |
|---|---|
| `finalize_tracker_failed.py` | Writes `summary.json` for a case whose flight passed every check but whose tracker schedule failed, with a status that marks the failure. |
| `finalize_hop_case.py` | Finishes a hop-distance case when only the fixed-mass re-solve missed the residual gate, by polishing it once and then writing the tracker check and summary. |
| `run_roundtrip.py` | Not a workflow. An empty source-identity anchor that the frozen core includes in its hash. Do not edit it. |

### `tests/`

| File | Covers |
|---|---|
| `test_dynamics.py`, `test_params.py`, `test_lambert.py` | Core dynamics (coast invariants, free fall, mass flow, CasADi/numpy agreement), parameter loading and unit checks, and the Lambert solver. |
| `test_roundtrip.py` | Runs the core round-trip optimization and checks convergence, touchdown conditions, the reserve factor, the sized-mass reproduction and independent re-propagation. This is the slowest test file. |
| `test_engine.py` | Engine classes, per-phase pools, burn-stage modes and the relight rule. |
| `test_sensor.py` | Hazard camera geometry. |
| `test_dem_crater.py` | Current DEM geometry and the physical-input identity hash of the current `a2200_08` case (update it deliberately when a physical input changes). |
| `test_active_cases.py` | Common comparison inputs and tracker update-age boundary acceptance, using the update-gap limit read from `mission.yaml`. |
| `test_descent_variant.py` | Descent settings: 2224 N thrust floor, touchdown thrust-to-weight limit, solve-side margin. |
| `test_star_tracker.py`, `test_tracker_dem.py`, `test_star_tracker_coupled.py` | Star-tracker numerics, DEM tracker checks and the coupled model. |
| `test_terrain_los.py`, `test_single_base_los.py` | Terrain line-of-sight geometry and single-station interval accounting. |
| `conftest.py` | Shared test fixtures. |

Some tests need the outputs that `run_case.py a2200_08` generates (see the Tests subsection of [Running each case](#running-each-case)).

### `data/` and `results/`

| Path | Contents |
|---|---|
| `data/terrain_corridor_30m.tif` | Native 30 m corridor DEM used for all terrain constraints and rays. |
| `data/ldem_16.img`, `data/ldem_16.lbl` | Global LOLA (Lunar Orbiter Laser Altimeter) elevation grid and its label, used for tracker rays outside the native footprint. |
| `data/terrain_source.json` | Source registration, citation and window of the corridor DEM (NASA LOLA data). |
| `results/a2200_08/`, `results/a2200_12/` | Only `sizing.npz` and `duals.npz`: the optimizer restart arrays. `run_case` has no cold start, so these are required. |

The GeoTIFF reader (`rasterio`) is installed from `requirements.txt`. All other results are generated locally and are not stored.

## Running each case

**All commands are run from the repository root** with the shared virtual environment, written as `.\.venv\Scripts\python.exe -B <script> ...`. The scripts find `Astro/` by themselves; no `PYTHONPATH` is needed. Every output folder named below is created locally when the script runs; none of them is stored in this repository.

**What a fresh clone contains.** Only two cases have their optimizer restart arrays stored: `Astro/results/a2200_08/{sizing.npz,duals.npz}` and `Astro/results/a2200_12/{sizing.npz,duals.npz}`. `run_case.py` has no cold start. For a case it needs `Astro/results/<case>/sizing.npz` plus `duals.npz`, or a `Astro/results/<case>/rejected_*/sizing.npz` restart (with its `duals.npz`). If neither exists it stops with the error "Sizing restart missing; restore the saved case arrays before running" (after creating an empty `results/<case>/` folder). If a `parameters.json` from an earlier run is present and the physical inputs have since changed, it stops and asks for `--retry`.

### Cases defined in `mission.yaml`

| Case | What it is | Command | Needs | Writes to | Runs from a fresh clone? |
|---|---|---|---|---|---|
| `a2200_08` | Selected design: 8 A2200 engines, 2,000 kg dry, 5,363.13 kg wet. | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py a2200_08` | The stored `sizing.npz` and `duals.npz`. | `Astro/results/a2200_08/`: `parameters.json`, `validation.json`, `trackers.json`, `summary.json`, `trajectory.npz`, `independent_flight.npz`, `phases.csv`, `REPORT.md`, the four plots (`trajectory_3d.png`, `flight_profiles.png`, `camera_visibility.png`, `tracker_coverage.png`), and `Astro/results/REPORT.md` (an overview of the cases). | Yes. |
| `a2200_12` | 12 A2200 engines, 5,128.67 kg wet. | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py a2200_12` | The stored `sizing.npz` and `duals.npz`. | `Astro/results/a2200_12/` (same files). | Yes. |
| `a2200_10` | 10 A2200 engines. Recorded result: converges but fails the hop-2 star-tracker check. | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py a2200_10` | `results/a2200_10/sizing.npz` and `duals.npz`, which are **not stored**. | `Astro/results/a2200_10/` | No. It stops with "Sizing restart missing". |
| `a2200_14` | 14 A2200 engines. Recorded result: converges but fails the hop-2 star-tracker check and the fixed-mass reproduction. | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py a2200_14` | `results/a2200_14/sizing.npz` and `duals.npz`, **not stored**. | `Astro/results/a2200_14/` | No. Same error. |
| `ar46_leros_rcs` | Mixed-engine backup: three AR46, four LEROS 4 and eight RCS thrusters. Recorded result: does not converge. | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py ar46_leros_rcs` | `results/ar46_leros_rcs/sizing.npz` and `duals.npz`, **not stored**. | `Astro/results/ar46_leros_rcs/` | No. Same error. |
| all five | Every case in turn. | `.\.venv\Scripts\python.exe -B Astro/scripts/run_case.py all` | All of the above. | All of the above. | No. It runs `a2200_08`, then stops at `a2200_10`. |

Options of `run_case.py`: `--report-only` regenerates the report and plots from an earlier full run of that case (it needs that run's `summary.json`, `validation.json`, `trackers.json` and `trajectory.npz`, so it cannot be the first command on a fresh clone); `--retry` moves an existing sizing attempt into a `rejected_<time>` folder and re-optimizes from its saved states and multipliers. Do not use `--retry` on the stored cases unless you intend to replace the stored arrays.

**What is missing for `a2200_10`, `a2200_14` and `ar46_leros_rcs`, and how it could be produced.** The missing file is a converged `sizing.npz` with its `duals.npz` for that case. No script in this repository creates a first restart array from scratch. The scripts that write a `sizing.npz` and `duals.npz` (`run_descent_variant.py`, `descent_ladder.py`, and `run_case.py` itself) all start from an existing seed `sizing.npz`; the recorded solutions for these cases came from continuation from earlier solutions that are not stored. Whether an `a2200_08` or `a2200_12` array is a usable seed for another engine count has not been tried here.

`run_case.py` normally reuses the saved sizing arrays, independently validates the flight and the reserve, checks the trackers and regenerates the report and plots. The recorded mixed-engine failure is displayed without automatically retrying it. Input changes require reoptimization; old validation does not certify changed inputs. The files written for an accepted case are read by GN and by the animation scripts.

### Studies and supporting scripts

"After `run_case a2200_08`" means after the first row above has been run once, so that `Astro/results/a2200_08/` holds the generated files.

| Study | Command | Needs | Writes to | Prerequisites available from a fresh clone? |
|---|---|---|---|---|
| Radio line of sight to the launch base | `.\.venv\Scripts\python.exe -B Astro/scripts/run_los.py` | `a2200_08` `summary.json`, `validation.json` (must show a validated flight) and `independent_flight.npz` | `Astro/results/los/` | Only after `run_case a2200_08`. |
| Direct-to-Earth line of sight | `.\.venv\Scripts\python.exe -B Astro/scripts/run_earth_los.py [case]` (default `a2200_08`) | `results/<case>/parameters.json` and `independent_flight.npz` | `Astro/results/earth_los/` | Only after `run_case a2200_08` (or `a2200_12`). |
| Delta-v losses | `.\.venv\Scripts\python.exe -B Astro/scripts/run_dv_losses.py [case]` (default `a2200_08`) | `results/<case>/parameters.json`, `trajectory.npz` and `summary.json` | `Astro/results/<case>/dv_losses.json` (inside the case folder) | Only after `run_case` of that case. |
| Tracker Sun/Earth sweep | `.\.venv\Scripts\python.exe -B Astro/scripts/run_tracker_sun_sweep.py` (add `--report-only` to redraw from `sweep.json`) | `results/a2200_08/sizing.npz` only | `Astro/results/tracker_sun_sweep/a2200_08/` | Yes. |
| Tracker mount reconstruction | `.\.venv\Scripts\python.exe -B Astro/scripts/reconstruct_tracker_mounts.py` (options `--case`, `--step`, `--fine-step`, `--grid-step`, `--random-pairs`, `--maxfev`, `--resume-grid`, `--report-only`, `--out`) | `a2200_08` `summary.json` (its saved hashes must match the current files), `trackers.json` (passed) and `sizing.npz`; it also fingerprints `GN/results/` files if they exist | `Astro/results/mount_reconstruction/a2200_08/` (`--out` must be a new folder under `results/`; it refuses to overwrite an existing `study.json`) | Only after `run_case a2200_08`. |
| Hop-distance demonstration | `.\.venv\Scripts\python.exe -B Astro/scripts/run_hop_distance.py [d050 d100 d150]` (or `--report-only`) | The solves need no stored array (built-in Lambert-based guess). The final report and plot step (also `--report-only`) reads `a2200_08` `summary.json` and `independent_flight.npz`. | `Astro/results/hop_distance/<case>/`, then `REPORT.md` and `comparison.png` in `Astro/results/hop_distance/` | The solves yes; the report step only after `run_case a2200_08`, otherwise the script ends with a file-not-found error after solving. The solves were not re-run against the current `mission.yaml` for this guide; the recorded numbers below come from earlier runs. |
| Design checks (dry mass -10 % and +10 %, doubled mesh) | `.\.venv\Scripts\python.exe -B Astro/scripts/run_design_checks.py [dry1800 dry2200 mesh_other2x]` (or `--report-only`) | Seed: stored `results/a2200_08/sizing.npz`. The report step reads `a2200_08` `summary.json` and `sizing.json`. | `Astro/results/design_checks/<variant>/` and `REPORT.md` | The variant solves yes. The final report step needs `results/a2200_08/sizing.json`, which `run_case.py` writes only when it solves the sizing itself, not when it loads the stored arrays; so on a fresh clone the report step stops with a file-not-found error after the variants are written. |
| Descent ladder (continuation to the `mission.yaml` descent settings, then the acceptance check) | `.\.venv\Scripts\python.exe -B Astro/scripts/descent_ladder.py a2200_12` (case, or `--seed <results folder>`, `--distance-km`, `--skip-floor`, `--accept-near`, `--depth`, `--wall`, `--tw`) | A warm-start `sizing.npz`: `results/<case>/sizing.npz`, or the `--seed` folder | `Astro/results/ladder_<case>/` (one folder per step plus `ladder.json`); with `--distance-km N`, `ladder_dNNN`; with `--skip-floor`, `_resume` is appended | Yes for `a2200_08` and `a2200_12` (convergence from the stored arrays was not re-run for this guide). Not for `a2200_10` and `a2200_14`. For the hop-distance use it needs `results/hop_distance/dNNN` from `run_hop_distance.py`. |
| One descent-variant solve | `.\.venv\Scripts\python.exe -B Astro/scripts/run_descent_variant.py TAG --seed a2200_08 --floor 2224 --tw 1.0 --obs 5 --vcap 5` | A seed folder holding `sizing.npz` | `Astro/results/TAG/` (`sizing.npz`, `duals.npz`, `diag.json`) | Yes, with `--seed a2200_08` or `--seed a2200_12`. |
| Acceptance check of a descent-variant run | `.\.venv\Scripts\python.exe -B Astro/scripts/check_descent_variant.py TAG --floor 2224 --tw 1.0 --obs 5 --vcap 5` (pass the same settings the run was solved with; `--case`, `--obs-nodes`, `--solve-pointing-rate`, `--distance-km`) | `results/TAG/sizing.npz` and `duals.npz` from `run_descent_variant.py` | `Astro/results/TAG/acceptance.json` | Only after a descent-variant run exists. |
| Finish a tracker-failed case | `.\.venv\Scripts\python.exe -B Astro/scripts/finalize_tracker_failed.py <case>` | `results/<case>/sizing.npz`, `validation.json` and `trackers.json` from a `run_case` attempt that failed the tracker check | `Astro/results/<case>/` (`summary.json`, `trajectory.npz`, `independent_flight.npz`) | No: no stored case fails its tracker check. |
| Finish a hop-distance case | `.\.venv\Scripts\python.exe -B Astro/scripts/finalize_hop_case.py d050 [d100 d150]` | `results/hop_distance/<case>/` from `run_hop_distance.py` | `Astro/results/hop_distance/<case>/`, then the report and plot | Only after `run_hop_distance.py` has run. |

`scripts/run_roundtrip.py` is not a workflow (see the file map).

### Tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider Astro/tests
```

Some tests need the files that `run_case.py a2200_08` generates, so run that first.
## Model and fixed assumptions

| Item | Active main mission |
|---|---|
| Launch | 85.5 S, 87.1 W; DEM elevation -2734.004134 m |
| Nominal destination | Drygalski floor, 78.826684042 S / 85.958520773 W; 202.402402 km away; DEM elevation -4818.455566 m |
| Dry mass | 2,000 kg |
| Divert / landing region | 150 m uprange from nominal destination / within 25 m of selected site; return within 25 m of launch |
| Landing limits | Downward speed 1.5 m/s, horizontal speed 0.5 m/s, thrust/body-up tilt 30 deg |
| Propellant reserve | Loaded propellant at least 1.1 times round-trip consumption |
| Departure / path clearance | Vertical rise to 50 m; ascent/coast/braking at least 50 m above DEM |
| Terminal descent | Existing 35 deg glide-slope cone relative to the chosen landing point |
| Decision gate | End of descending observation, 150–1500 m above DEM; outbound gate above nominal target |
| Observation | At least 5 s acquisition (placeholder sensitivity-study value; it was 1 s) + 0.1 s processing, 5 m minimum AGL, 0.1–5 m/s downward (placeholder; it was 0.1–100 m/s) and at most 100 m/s horizontal |
| Hazard camera | Fixed downward body mount, usable 90 x 90 deg FOV, 10 km range, both nominal and chosen touchdown points visible; DEM rays checked |
| Attitude proxy | Thrust-aligned body-up; projected-downrange roll and inertial endpoint SLERP during coast; vehicle thrust-direction rate 6 deg/s (star-tracker rating), every phase |
| Two trackers | Independent top mounts, 20 deg full circular FOV; Sun/Earth exclusion 23.5 deg, full cone clear of terrain, ideal top-plane obstruction proxy |
| Tracker updates | Either head, ascent/coast only, maximum update/estimate age 5 s including block boundaries; initialized zero-age estimate per block; instantaneous acquisition/switching |
| Tracker rate | At valid update events, full inertial body rate at most 6 deg/s |

AGL means above ground level. SLERP is spherical linear interpolation of attitude.

Existing sensor operating limits and synthetic celestial geometry remain modeling assumptions, not verified hardware capability. Sun/Earth directions are fixed inertially at the configured vectors; there is no real-ephemeris or star-catalog simulation. Mount elevations are 72.66012644 / 54.93565637 deg above body XY, azimuths 0 / 180 deg, shared between hops. Flat-site contact uses radial local up; slope, roughness, landing gear, impact loads, shutdown transients and attitude torque dynamics are outside this model.

Each hop is vertical rise, staged ascent, engines-off coast, staged braking, descending observation, and terminal descent. Ascent/rise form one continuous burn; braking/observation/terminal form the other. The return starts at the actual outbound landing point, at rest with the same remaining mass; surface-stay time and consumption are omitted. Four A2200s remain during observation/landing, with N -> (N+4)/2 -> 4 shutdown stages on both powered arcs. Engines remain commanded through the terminal endpoint; no cutoff/impact model is added.

A2200 inputs remain 2224 N per engine, 318 s constant Isp, 50–100% shared throttle and 1/6 per second throttle-fraction slew. Mixed-engine hardware retains three AR46 (7060 N / 277 s, non-pulsing, 66 s continuous-burn cap), four standard LEROS 4 (1000 N / 321 s, mean pulsing) and eight generic axial RCS (200 N / 317 s, mean pulsing). Pulse frequency/minimum impulse is unspecified; actual pulse ignition counts are not inferred. Engine/feed hardware mass is not separately adjusted between architectures.

The unchanged `hopper/` core uses Moon-fixed position/velocity with central gravity, lunar rotation, Coriolis/centrifugal terms, finite thrust and engine-class mass flow. States are position (3), velocity (3) and mass. Powered controls are thrust vector (3) and magnitude Gamma; vector norm equals Gamma at nodes. Segment durations and initial wet mass are optimized, including engine shutdown timing. Engine counts/stage order and tracker mounts are configured. Hermite–Simpson collocation, the staged mesh and IPOPT remain.

`mission/terrain.py` adds the same native DEM interpolation, domain/clearance constraints, midpoint checks and observation terrain rays as the accepted main case. `mission/validation.py` independently integrates each linear-control interval and detects first descending DEM contact. Flight, camera, hardware, reserve reproduction and terrain-aware tracker schedules must pass before final export. Terrain cone/ray checks are sampled and are not a continuous terrain-uncertainty certificate.

## Optimization and acceptance

**Active workflow (user decision, 2026-10-28): optimize the trajectory with DEM constraints and check star trackers separately on the saved flight.** Use `scripts/run_case.py` for the DEM cases and `scripts/run_hop_distance.py` for the distance study. Tracker visibility, body rate and update cadence remain post-processing checks; they do not constrain or reshape the optimized trajectory. Existing accepted DEM results remain the working baseline. The alternative-solver trials and the DEM + tracker joint solve were stopped at the user's request; no new trajectory was accepted.

An accepted case must pass all of these: independent flight propagation and contact checks, the hazard camera check, the hardware check, the thrust-discard check, a fixed-mass reproduction of the reserve, and the terrain-aware star-tracker schedule. A rejected solve is diagnostic only and is never reported as accepted mission performance.

At the 6 deg/s rate limit the ascent mesh is 540 intervals per hop (it was 180). The 180-interval solve failed the independent return touchdown check (0.5036 m/s) because the rate-limited pitch-over puts corners in the thrust history. Sizing jobs at this size need a longer IPOPT wall time (1,500 s used).

The 6 deg/s main model has 21,537 decision variables and 39,998 constraints (15 deg/s mesh: 13,617 / 26,318). These are nominal local results, without a global-optimality proof or a formal mesh-error bound. Integrated finite-burn delta-v minus Lambert includes gravity, steering and trajectory differences; it is not pure gravity loss. The mesh and dry-mass study numbers are in the [history section](#recorded-results-and-history).

## Descent-thrust variant (opt-in, diagnostic)

Since 2026-11-04 these settings are on in `mission.yaml` and define the main `a2200_08` case:

- `descent_variant` section: 2224 N thrust floor, touchdown thrust/weight <= 1.0, 5 deg/s solve-side pointing margin.
- `common.sensor`: 5 s observation and a 5 m/s vertical descent-speed cap.
- `common.solver.nodes.observation`: 36 nodes.

The main case was regenerated through `run_case` (ALL CHECKS PASSED, 5,363.13 kg); the stored `a2200_08` restart arrays are that solution. The previous 5,309.64 kg case was archived on the author's machine as `results/a2200_08_baseline_5309kg`; that folder is historical and is not stored in this repository.

How the settings work:

- Setting a value to null turns it off.
- `thrust_floor_n` lowers the thrust floor of `floor_phases` (observation, terminal) on both hops in `mission.config.parameters`. It never raises a floor, and the stage maximum is kept.
- `touchdown_thrust_to_weight_max` adds the constraint thrust / local weight <= value at the last node of each hop in `GroundProblem`.
- `solve_pointing_rate_deg_s` is a numerical margin: the optimizer caps the observation pointing rate at this value, while the independent camera check still uses the real 6 deg/s.
- The 2224 N floor is a user-specified placeholder. With four A2200s it means 25 % throttle (the baseline floor is 50 %, 4448 N). It is not a verified engine capability. The 5 s observation and 5 m/s vertical cap are the sensitivity-study values of an earlier seed run named `A05` (a historical run that is not stored in this repository).

Tools:

- `scripts/run_descent_variant.py TAG --seed <results folder> --floor 2224 --tw 1.0 --obs 5 --vcap 5` solves a diagnostic run into `results/TAG/` with no acceptance checks. In this repository the seed folders that exist are `a2200_08` and `a2200_12`; exact commands are in [Running each case](#running-each-case).
- `scripts/check_descent_variant.py` applies the same checks as `run_case` to such a run.
- `scripts/descent_ladder.py CASE` automates the continuation: floor only, then T/W 2.0 ... 1.0 at 12 observation nodes, then 36 nodes, then the 5 deg/s margin. A failed step is halved up to 3 times. It runs the acceptance check at the end and writes only `results/ladder_CASE/`.
- `run_descent_variant.py` and `check_descent_variant.py` take `--case`.
- A jump straight to T/W <= 1 from the earlier `A05` seed (not stored) did not converge (junk iterate, no `Solve_Succeeded`). Seed from the floor-only solution and lower the limit in steps (2.0, 1.7, 1.4, 1.2, 1.1, 1.0).

Stated assumptions of the final descent-variant result (recorded run `results/dv_n36p5`, not stored; 5,363.13 kg, 2026-11-04):

1. Thrust below 4448 N is read as engines 3 and 4 shut off and engines 1 and 2 at 50–100 % (2224 N = two engines at 50 %). This burns the same propellant as the model's four engines at 25 %, because mass flow is thrust over exhaust velocity with no per-engine offset.
2. The throttle slew stays at 1/6 of the four-engine capacity (1483 N/s) everywhere, although two engines could only slew 741 N/s. The solution exceeds 741 N/s while at or below 4448 N (0.7 s per observation phase, 5.5 s hop-1 and 7.4 s hop-2 terminal).
3. Engine shutdown and relight are not modeled.
4. The hardware check still fails on its hard-coded 50 % A2200 throttle floor and was left unchanged.

Acceptance of the final run (`scripts/check_descent_variant.py`, same checks as `run_case`; recorded run `results/dv_n36p5`, observation mesh 36 nodes, solved with a 5 deg/s observation pointing margin and checked against the real 6 deg/s): flight, camera, thrust-discard, fixed-mass reserve (1.1000) and star-tracker schedule pass. The hardware check fails because it hard-codes the A2200 50 % throttle floor and this run flies at 25 %; all other hardware lines pass (firing <= 1000 s, longest burn 186 s, 4 starts, slew 1/6 per s). The 12-node observation mesh fails the camera check (hop-2 pointing rate 7.15 deg/s between nodes; 36 nodes gives 6.38 deg/s), so refine the mesh and solve with a margin.

Editing `mission.yaml` or `mission/*.py` changes the source hashes recorded in the saved `a2200_08` summary, so GN will not load that case until it is re-run.

## Status of each case

Current status (2026-11-04, with the descent-variant settings): **8 A2200 accepted, 5,363.13 kg; 12 A2200 accepted, 5,128.67 kg; 10 and 14 A2200 converge but fail the hop-2 tracker check.** Because `common` in `mission.yaml` is shared, the other cases' older saved results no longer matched it after the change, so they were handled as follows.

| Case | Status (2026-11-04) | Wet mass (kg) | Notes |
|---|---|---:|---|
| `a2200_08` | Accepted, ALL CHECKS PASSED | 5,363.13 | Selected design. Regenerated through `run_case`. |
| `a2200_12` | Accepted, ALL CHECKS PASSED | 5,128.67 | Re-run through the descent ladder and `run_case`. Tracker gap now 3.0 s. The old 5,082.61 kg case was archived on the author's machine as `results/a2200_12_baseline_oct28` (historical, not stored). |
| `a2200_10` | Converges but fails acceptance; recorded as is | 5,214.71 | Fails the hop-2 star-tracker schedule: max update gap 5.99 s at t = 651–657 s against the 5 s limit. Needed floor halvings (3892 N before 3336 N and 2224 N). |
| `a2200_14` | Converges but fails acceptance; recorded as is | 5,072.97 | Fails the hop-2 star-tracker schedule at t = 630 s, and fails the fixed-mass reproduction on a solver residual (primal 1.8e-6; mass and reserve ratio themselves are fine). Needed one T/W halving (1.3 before 1.2). |
| `ar46_leros_rcs` | Not re-run (user decision); keeps its stale not-converged result | — | Mixed-engine backup: three AR46, four LEROS 4 and eight RCS thrusters. Does not converge; no failed-iterate flight performance is reported. |

The `a2200_10` and `a2200_14` solutions were not saved into their case folders, which on the author's machine still held the stale 2026-10-28 results. The new solutions were in `results/ladder_a2200_10/13_margin` and `results/ladder_a2200_14/11_margin` (historical runs, not stored). The `a2200_12` ladder needed one T/W halving (1.85 before 1.7).

Earlier status (6 deg/s, 5 s gap, 2026-10-28, before the descent variant): 8 A2200 5,309.64 kg wet; 12 A2200 5,082.61 kg. 10 A2200: flight, camera, hardware and reserve checks pass, but the worst tracker update gap is 6.0 s > 5 s (status `flight_validated_tracker_gap_exceeded`). Mixed engines: does not converge. Earlier 15 deg/s results were archived on the author's machine as `results/archive_tilt15/` (historical, not stored).

The main-design line of sight (LOS), produced by `run_los.py`, covers the original launch base only, the actual DEM flight and no added descent or surface-stay duration.

## Inactive joint-solver experiment

An experimental workflow that coupled the star-tracker update events to the trajectory optimization (a "DEM + solver-coupled tracker" solve) was stopped at the user's request, and no trajectory from it was accepted. It is not part of the active workflow, and **there is no launcher for it in this repository**. Only its solver modules remain, because saved flights hash every file in `mission/` and `hopper/`: `mission/tracker_solver.py` and `hopper/star_tracker_coupled.py`, with their tests (`tests/test_tracker_dem.py` and `tests/test_star_tracker_coupled.py`). Its numerical settings stay in `mission.yaml:tracker_solver`.

What the modules model: the common installed mount angles are fixed, and explicit labeled updates are added during ascent and coast on both hops. Event times follow the optimized phase durations; the first, intermediate and final estimate ages, the Sun/Earth exclusions and the full inertial rate limit are constrained. Terrain cones use sampled rays through the native DEM (with the global LOLA grid outside the native footprint), so this is sampled terrain coverage and local trajectory optimization, not continuous clearance or global optimality. No star catalog, real ephemerides, new sensor limits or attitude torque dynamics are added. The active workflow instead optimizes the trajectory with DEM constraints and checks the star trackers afterwards on the saved flight; GN reads the validated `a2200_08` flight. The dated history of the experiment is in the [history section](#recorded-results-and-history).
## Recorded results and history

The numbers below were produced locally on earlier runs. The result folders they mention (for example `results/hop_distance_6degs`, `results/ladder_*` and `results/design_checks`) are historical runs that are not stored in this repository; the figures are kept here as the team's record. The scripts named in this section write their own folders when run (see [Running each case](#running-each-case)).

### Hop-distance demonstration

`scripts/run_hop_distance.py` (local output `results/hop_distance/`) flies the 8 A2200 design 50, 100 and 150 km from the same launch base toward Drygalski with the same inputs and acceptance checks; only the destination changes. All three validated.

- Archived 15 deg/s / 2 s results, kept by user decision. A 6 deg/s spot check of 50 km (`results/hop_distance_6degs/d050/`) also validated (3,223.76 kg wet).
- **2026-11-04 re-run with the descent-variant settings** (2224 N floor, T/W <= 1, 36 observation nodes, 5 deg/s margin): run with `scripts/descent_ladder.py a2200_08 --distance-km N --seed hop_distance/dNNN`, results in `results/ladder_d050`, `ladder_d100_resume` and `ladder_d150`, last step folder `*_margin`. The first `ladder_d100` stopped at T/W 2.75 on a primal residual of 3e-6; it was resumed with `--skip-floor --accept-near --depth 5`. Results are recorded as is, with no summary or plots generated.

| Hop distance | Wet mass (kg) | Acceptance result |
|---|---:|---|
| 50 km | 3,253.21 | Passes everything except the fixed-mass reproduction, which fails only on the strict solver residual (primal 1.2e-6; reserve ratio 1.1000 and mass difference -4e-5 kg are fine; tracker gaps 3.0 / 3.99 s). |
| 100 km | 3,961.16 | Passes all checks (tracker gaps 4.99 s hop 1, 4.00 s hop 2). |
| 150 km | 4,638.59 | Passes everything except the hop-2 star-tracker schedule (fails at t = 582 s; hop-1 gap 3.0 s). |
| 202 km (`a2200_08`) | 5,363.13 | Accepted main case. |

The 150 km failure pattern is the same as `a2200_10` and `a2200_14`.

### Star-tracker mounting reconstruction (2026-11-08)

`scripts/reconstruct_tracker_mounts.py` evaluates alternative mounts on the saved `a2200_08` flight without changing the trajectory, configuration or GN results. The study uses availability as its stated objective; it does not recover the deleted historical optimization. Its 5 deg elevation grid (azimuths fixed at 0/180 deg) tested 289 pairs: 283 pass the sampled 5 s update-age limit and 271 tie for best coarse-grid coverage. The retained 72.6601/54.9357 deg pair reaches the rate-only coverage upper bound at the 0.25 s fine grid, so further four-angle refinement was unnecessary for that sampled objective. Half-step checking gives 99.3016% either-head availability and 0.7499 s maximum sampled age during ascent/coast; the tested 72/55 deg approximation also ties at 0.25 s. This supports retaining the installation, not a unique historical optimum or a whole-flight GN claim.

The script writes its report (`REPORT.md`), the plots `mounting_trade.png` and `angle_vs_coverage.png`, and `study.json` / `input_hashes.json` to `results/mount_reconstruction/a2200_08/` when run (generated locally, not stored); 66 protected files had unchanged SHA-256 hashes. Use `--report-only` to regenerate the figure and explanation from saved study results; use a new `--out` for a fresh study.

### Mesh and dry-mass checks

Ascent-mesh study (recorded as `results/design_checks/ascent_mesh_series.json`; that file is not stored and `run_design_checks.py` does not produce it): 180 / 360 / 540 / 720 intervals give 5,302.23 (fails touchdown) / 5,307.44 / 5,309.64 / 5,310.84 kg, with an extrapolated limit of about 5,316 kg. This study predates the descent variant; its 540-interval value is the earlier 5,309.64 kg main case.

Dry-mass checks: 1,800 kg dry -> 4,704.10 kg wet, 2,200 kg dry -> 5,942.46 kg wet, both validated. These are nominal local results, without a global-optimality proof or formal mesh-error bound.

### Descent-variant ladder results (2026-11-04)

Unvalidated diagnostic solves, wet mass in kg:

| Run | Wet mass (kg) |
|---|---:|
| `A05` seed | 5,358.01 |
| Floor 2224 N only | 5,353.85 |
| Plus T/W limit 2.0 / 1.7 / 1.4 / 1.2 / 1.1 / 1.0 (`results/dv_c1.0`) | 5,354.26 / 5,355.18 / 5,356.88 / 5,359.04 / 5,360.45 / 5,362.11 |
| Final run `results/dv_n36p5` (36 observation nodes, 5 deg/s margin) | 5,363.13 (accepted, see above) |

Without the limit, hop 1 touches down at T/W 1.58 and hop 2 at 2.37. At the 1.0 limit both end at 1.00 with 1.58 m/s ground speed.

### Joint-solver experiment (2026-10-28)

The combined DEM and solver-coupled tracker workflow was implemented, but its trajectory optimization validation stayed pending. Initial SciPy `_pava_pybind` import blocks stopped reproducing with Smart App Control still enabled. The full Astro suite then recorded 151 passed and 16 optimization-fixture errors; all 9 tracker tests passed in the subsequent focused rerun, including a new zero-age boundary regression. The pilot exposed an identically zero endpoint-age inequality, which was omitted because CasADi requires symbolic constraints. The IPOPT preflight stayed unavailable: Windows CodeIntegrity events identified its dependency `casadi/libcasadi-tp-openblas.dll` as blocked by Smart App Control, and the unchanged optimizer tests met the same dependency block. The former launcher's `--check-environment` option checked IPOPT availability without loading terrain or solving. No security settings were changed and no tracker-coupled trajectory was accepted. Acceptance for that workflow would have required the existing independent flight, landing, camera, hardware and reserve checks plus DEM visibility and cadence validation of the exact assigned tracker events against chained independent flight states, with the same tracker constraints in the fixed-mass reproduction.
