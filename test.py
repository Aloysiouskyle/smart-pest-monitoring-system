from pathlib import Path

from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
model_path = project_dir / "runs" / "detect" / "pest_detection_model-6" / "weights" / "best.pt"
image_path = (
    project_dir
    / "dataset"
    / "fall armyworm.v2i.yolov8"
    / "valid"
    / "images"
    / "0A5E7104-FF40-49E9-95CD-EA530C6D5519-12_11_44_jpeg_jpg.rf.efa45fd6db9490826a7f5e705e54b021.jpg"
)

model = YOLO(model_path)
results = model(image_path)

results[0].show()