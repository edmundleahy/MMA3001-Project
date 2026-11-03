# Forecasting Indoor Temperature from a Monash Environmental Sensor

MMA3001 individual project, using teaching-team **Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data**.

**Finding:** one hour ahead on the selected sensor, a three-input linear regression beats simply assuming the temperature stays the same, but only slightly on typical error: test MAE is 0.365 °C against 0.380 °C. It helps more with large errors, with RMSE at 0.607 °C against 0.697 °C. The hours from 7 to 9 am hold about 40% of persistence's total error, because a sharp temperature rise around 8 am is forecast late by both methods, and the regression's gain comes entirely from those hours.

## Research question
How accurately can future indoor temperature be predicted from recent observations of a single environmental sensor, and does a lagged linear regression model beat a persistence baseline `T̂(t+H) = T(t)`?

| | |
|---|---|
| **Input** | The latest three 20-minute temperature values from one sensor, `T(t)`, `T(t−20 min)`, `T(t−40 min)`, in °C |
| **Output** | Predicted temperature in °C at `t + 60 min` |
| **Intended use** | Short-term monitoring of one sensor's location. It is not a control system and makes no claim about energy use or comfort |
| **Invalid or missing input** | If any input or target value is missing, that example is excluded and counted. Gaps are never interpolated |
| **Methods** | Persistence baseline; `scikit-learn` `LinearRegression` |
| **Evaluation** | Chronological 60/20/20 split with purged boundaries; MAE and RMSE in °C; sensitivity to lags, horizon and time step; runtime, memory and FLOPS |

## Results
Fitted on the training period only. Settings were chosen on `6012002000869` using validation data, then applied unchanged to both sensors.

| Test period | Forecasts | Persistence MAE / RMSE (°C) | Regression MAE / RMSE (°C) | MAE gain (95% interval, °C) |
|---|---:|---:|---:|---:|
| `6012002000869`, 24 May – 14 Jun 2024 | 1,505 | 0.380 / 0.697 | 0.365 / 0.607 | 0.015 (0.002 to 0.030) |
| `6012002000326`, 9 Mar – 22 Apr 2026 (location unverified) | 3,185 | 0.444 / 0.605 | 0.348 / 0.508 | 0.096 (0.073 to 0.121) |

The interval comes from a bootstrap that resamples whole local days, because errors close together in time are correlated.

The selected model, fitted on `6012002000869`:

`T̂(t+60 min) = 2.807 + 1.650·T(t) − 0.520·T(t−20 min) − 0.251·T(t−40 min)`

### What this means
- **The MAE gain on the selected sensor is small.** The interval excludes zero, but 0.015 °C is far below the 0.25 °C median step between recorded temperature values.
- **Errors are concentrated in time.** At 8 am local time, MAE is 2.42 °C for persistence and 1.87 °C for regression. Averaged across the hours outside 7–9 am, persistence is slightly better (0.258 °C against 0.264 °C), and it is better in every hour from 1 pm to midnight. See `fig_error_by_hour.png`.
- **The second sensor shows a larger gain**, 21.6% in MAE, but its location is not in the supplied workbook, so it cannot be confirmed as an indoor space.

### Sensitivity (validation period, `6012002000869`)
Persistence MAE is 0.375 °C at the 60-minute horizon.

| Lagged inputs | Regression MAE (°C) | Condition number |
|---:|---:|---:|
| 1 | 0.398 | 1 |
| **3 (selected)** | 0.349 | 20 |
| 6 | 0.349 | 31 |
| 12 | 0.348 | 43 |

| Horizon, 3 inputs | Persistence MAE (°C) | Regression MAE (°C) | MAE improvement |
|---:|---:|---:|---:|
| 20 min | 0.138 | 0.137 | 0.4% |
| **60 min** | 0.375 | 0.349 | 7.0% |
| 180 min | 0.932 | 0.791 | 15.2% |

| Time step (about 1 hour of history) | Training examples | Regression MAE (°C) |
|---:|---:|---:|
| 10 min | 3,585 | 0.381 |
| **20 min** | 4,553 | 0.349 |
| 30 min | 3,035 | 0.346 |

**Selection rule:** the fewest inputs whose validation MAE is within 0.01 °C of the best. This replaced an initial "lowest MAE" rule, which chose 12 inputs for a 0.001 °C gain. The change was made after seeing validation results and before any test-period scoring.

### Performance
On the 4,544 training rows, on one Windows laptop: fitting takes about 1.4 ms and predicting all rows about 0.6 ms, against 0.02 ms for persistence. Peak Python allocation while fitting is about 0.4 MB. The textbook estimates are about 82,000 FLOPS to fit and 6 per prediction. Details are in `performance.json` and `profile_select.txt`.

## Data
- File: `5EnvSensor_MayToDec2024_180kRows.zip`, supplied by the teaching team through Moodle. **It is not included in this repository.** Copy it to `data/raw/`, which is git-ignored.
- Measurements are stored as a JSON array inside the CSV column `jsondata`. The pipeline uses each reading's **event time** (`values[].time`, Unix seconds UTC), not `createdate`, which is ingestion metadata.
- The archive marked "Example Data (Do not use)" is not used anywhere in this project.

### Audit findings
- The 179,447 rows hold only **42,843 unique temperature readings**. Most rows repeat an old reading.
- Only two of the five sensors are usable: `6012002000227`, `6012002000241` and `6012002000777` have 1 or 2 unique readings each.
- `6012002000869` stopped producing new readings on 14 June 2024, yet its last reading was re-sent 15,431 times, with `createdate` values running until October 2024.
- Despite the filename, event times run from May 2023 to April 2026. Only the May–June 2024 part of `6012002000869` falls inside the named window.
- There were no parse errors and no conflicting duplicates.

### Sensors
| Sensor | Location in workbook | Period used (local) | Grid slots filled | Role |
|---|---|---|---:|---|
| `6012002000869` | Building 60, G.38, "Compactus_upper" (EMS107) | 29 Feb – 14 Jun 2024 | 99.99% | Primary. The only one of the five sensors with a listed location |
| `6012002000326` | Not listed | 11 Sep 2025 – 22 Apr 2026 | 97.1% | Independent check only |

Readings of `6012002000326` before 11 September 2025 are excluded because they are isolated from the continuous record, by gaps of 636 and 4.1 days.

## Setup (Windows PowerShell)
Tested on Python 3.10.1. Exact package versions are in `results/pip_freeze.txt`.
```powershell
cd "D:\Projects\MMA3001 Project"   # adjust to your clone location
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Reproduce
```powershell
# 1. Unit tests (synthetic data only) and test report
python -m pytest -v --junitxml=results/test_report.xml | Out-File results\test_report.txt -Encoding utf8

# 2. Data audit: streams the ZIP and writes data/processed/readings.csv
python -m src.main audit

# 3. Choose settings using training and validation data only
python -m src.main select
python -m src.main profile

# 4. Evaluate on the test periods
python -m src.main final --reproduce
python -m src.main final --sensor 6012002000326 --settings-from 6012002000869 --reproduce

# 5. HTML code documentation
python -m pdoc src -o docs
```
`final` refuses to overwrite an existing test result unless given `--reproduce`, which re-reads the same locked settings. Test metrics are deterministic. Runtime figures vary by machine.

Open `docs/index.html` for the code documentation. Outputs are in `results/audit/` and `results/forecast/<sensor>/`.

## Method
- **Grid:** each 20-minute slot holds the latest reading at or before the slot's end time, so its value was already known at that time. Empty slots stay empty.
- **Why 20 minutes:** readings arrive about every 10 minutes, but about 1 in 10 is missed. A 10-minute grid loses examples, while a 20-minute grid is almost fully filled.
- **Split:** 60/20/20 by time. An example is kept only if its whole window, from oldest input to target, lies inside one partition, and the rest are counted as purged. A random split would place near-identical neighbouring readings in both training and test, which would make the error look better than it is.
- **Scaling:** none. All inputs are in °C and scaling does not change an ordinary least squares prediction.
- **Duplicates:** keyed on `(sensorid, variable, event time)`. Identical repeats collapse to one reading and conflicting values would take the median. The audit found no conflicts.
- **Time:** ordering and splitting use UTC instants. Melbourne local time, with AEST and AEDT handled by the time zone database, is used only for hour-of-day analysis and plot axes.

## Verification and validation
- **Verification, meaning the code does what it specifies:** 63 unit tests on synthetic data, reported in `results/test_report.txt`. They cover parsing, time zones, duplicates, target alignment across gaps, features not changing when future values change, no slot shared between partitions, hand-calculated metrics, recovery of known coefficients, agreement with an independent least-squares solution, and repeatable results. One test rewrites the test period by +50 °C and confirms that selection does not change.
- **Validation, meaning predictions hold on unseen real data:** chronological holdout against the persistence baseline on each sensor, plus a second sensor that played no part in selecting the settings.

## Limitations
- Results apply to these sensors and periods. They are not claimed to transfer to other rooms, buildings, seasons or HVAC regimes.
- The selected sensor covers about 3.5 months of one season, and its test period is 22 days. A compactus may be a low-activity space.
- The cause of the 8 am rise cannot be determined from this data. It may be related to building operation, but that is not established here.
- A 20-minute slot holds the latest reading in that slot, so the true gap between input and target can differ slightly from a nominal 60 minutes.
- Temperature is recorded in steps of about 0.25 °C, which limits how much a small error difference can mean.
- The bootstrap treats days as independent, which is an approximation.
- Only past temperature is used. Time of day, humidity and CO2 were not tested as inputs, so the morning error might be reducible.
- The model is linear and fixed after training. Its coefficients are not updated as conditions change.

## Repository layout
```
src/data_processing.py   streaming ZIP → CSV → JSON parser, deduplication
src/analysis.py          coverage audit tables and figures
src/features.py          grid, lag features, target, chronological split
src/models.py            persistence, regression, metrics, cost estimates
src/evaluate.py          select and final stages, sensitivity, bootstrap
src/plots.py             report figures
src/main.py              command-line interface
tests/                   pytest unit tests (synthetic data)
data/raw, data/processed git-ignored data folders
results/                 audit, metrics, figures, profile and test report
docs/                    generated HTML documentation (pdoc)
report/                  written report and AI usage log
```

## AI use
AI assistance (Claude, with ChatGPT used earlier for project planning) was used for the code, tests and this README. See `report/AI_USAGE_LOG.md` for the disclosure and reflection.

## Licence
The code is released under the MIT licence in `LICENSE`. That licence covers the original source code only and **does not** cover the Monash sensor data.
