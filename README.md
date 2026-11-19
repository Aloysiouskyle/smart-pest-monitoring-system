# Smart Pest Monitoring System Using Distributed AI Agents in Uganda

A final-year Computer Science project exploring offline-capable, multi-agent pest monitoring for smallholder maize farmers in Uganda.

## Current prototype

- YOLOv8 Nano object detection for maize pests and plant conditions.
- Streamlit image-upload dashboard with detection bounding boxes and confidence scores.
- Grad-CAM visualization for model interpretation (requires the matching PyTorch model weights).
- Basic English advisory text based on detected classes.

## Planned SYE3104 module (18 November – 2 December 2026)

- WP 2.1 synthetic microclimate telemetry (implemented below).
- WP 2.2 standalone rule-based assessment of telemetry and visual detections.
- English/Luganda farmer SMS templates.
- Integration tests, user acceptance testing, and project management evidence.

Dashboard integration of these modules, bilingual advice, and field validation are planned work.

## WP 2.1 microclimate simulator

`microclimate_sim.py` uses the Python standard library. From the repository root:

```python
from microclimate_sim import export_scenario, generate_telemetry

normal = generate_telemetry("normal", seed=42)
higher_risk = generate_telemetry("higher_risk", seed=42)
print(normal[0])

# Optional exports. Repeating an export replaces that scenario's file.
export_scenario("normal", "csv", seed=42)        # data/generated/normal.csv
export_scenario("higher_risk", "json", seed=42)  # data/generated/higher_risk.json
```

Each call generates 72 hourly readings, starting at midnight on 18 November 2026 in Uganda (UTC+03:00) by default. Pass a timezone-aware `datetime` as `start` to choose another whole hour. The same scenario, seed, and start reproduce the same records in memory and in either export format.

| Synthetic scenario | Temperature (°C) | Relative humidity (%) | Rainfall (mm/hour) | Soil moisture (%) |
|---|---:|---:|---:|---:|
| Normal maize field (`normal`) | 19–29 | 55–85 | 0–4 | 34–42 |
| Simulated higher Fall Armyworm risk (`higher_risk`) | 22–32 | 65–95 | 0–6 | 42–52 |

Temperature peaks around mid-afternoon, humidity rises overnight, and occasional rain increases the simulated soil moisture for that hour. The ranges and relationships are illustrative choices for testing. The scenario labels carry no validated outbreak thresholds or predictive meaning. The 72 hours describe the generated telemetry window rather than a forecast horizon.

Run the simulator tests with `python -m unittest discover -s tests -p test_microclimate_sim.py -v`.

## WP 2.2 Fall Armyworm forecasting agent

`assess_risk()` evaluates two demonstration rules. Each can contribute one point.

| Evidence | Default rule for one point |
|---|---|
| Simulated microclimate | At least 18 of 72 hours have **both** temperature ≥25 °C and relative humidity ≥72%. Matching hours can occur anywhere in the window. |
| Submitted image | At least one Fall Armyworm egg, frass, larva, or larval-damage box has YOLO confidence ≥0.50. Only the strongest Fall Armyworm score counts. |

Zero points gives **Low**, one gives **Moderate**, and two gives **High**. Rainfall and soil moisture are validated as part of every record. Only temperature and humidity affect the climate point. Healthy maize and maize streak disease are accepted detector classes and contribute zero points.

```python
from forecasting_agent import RiskRules, assess_risk
from microclimate_sim import generate_telemetry

rows = generate_telemetry("higher_risk", seed=42)
boxes = {"fall-armyworm-larval-damage": [0.92945], "maize-streak-disease": [0.10267]}
result = assess_risk(rows, boxes)  # Grouped scores in the dashboard's format
print(result.risk_level, result.contributing_factors, result.image_status)
print(result.limitation)

# Optionally override inclusive cutoffs for another demonstration.
custom = assess_risk(rows, boxes, rules=RiskRules(min_matching_hours=24))
```

Pass `None` for an unsubmitted image (`no_image`) or `{}` for a processed image with zero boxes (`zero_detections`). Both contribute zero visual points. A nonempty class-to-scores mapping gives `detections_present`. A Low result can therefore reflect missing visual evidence. Read the image status alongside the level.

The agent requires 72 consecutive timezone-aware, whole-hour observations in order. All four measurements must lie within the combined WP 2.1 simulation bounds shown above. Incomplete telemetry, unknown labels, empty per-class score lists, and invalid scores raise `ValueError`. YOLO confidence describes a box in the current photo. The thresholds and categories are illustrative and have no validated relationship to infestation probability or outbreaks in the next 48–72 hours.

Run both work packages' automated tests with `python -m unittest discover -s tests -v`.

## Source files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `inference.py` | YOLO inference and Grad-CAM helper |
| `train.py` | Train model on a local dataset |
| `export.py` | Export trained model to ONNX |
| `test.py` | Manual single-image inference demo (not an automated test suite) |
| `data.yaml` | Local dataset configuration |
| `requirements.txt` | Application dependencies |

## Local setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard requires trained weights at one of the locations referenced in `app.py`, such as `runs/detect/pest_detection_model-6/weights/best.pt`. These files are excluded from version control. The dataset specified in `data.yaml` must also be obtained separately before training. Additional dependencies, including PyTorch, may be resolved by the Ultralytics package; GPU configurations require a suitable PyTorch/CUDA installation.

## Notes

- Benchmarks reported on a development laptop should not be interpreted as Raspberry Pi 4 benchmarks.
- Simulated microclimate scenarios do not establish real-world forecasting accuracy.
- Review all files before publishing the repository publicly; datasets and model weights may have their own license conditions.

## Project status

Development prototype. Further forecasting, advisory, physical deployment and field validation work is ongoing.
