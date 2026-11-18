# Smart Pest Monitoring System Using Distributed AI Agents in Uganda

A final-year Computer Science project exploring offline-capable, multi-agent pest monitoring for smallholder maize farmers in Uganda.

## Current prototype

- YOLOv8 Nano object detection for maize pests and plant conditions.
- Streamlit image-upload dashboard with detection bounding boxes and confidence scores.
- Grad-CAM visualization for model interpretation (requires the matching PyTorch model weights).
- Basic English advisory text based on detected classes.

## Planned SYE3104 module (18 November – 2 December 2026)

- Simulated 72-hour microclimate measurements.
- Rule-based risk fusion of microclimate observations and visual detections.
- English/Luganda farmer SMS templates.
- Integration tests, user acceptance testing, and project management evidence.

These planned features are **not yet implemented** in this initial code snapshot.

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
