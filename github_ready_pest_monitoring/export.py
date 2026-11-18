from ultralytics import YOLO

# 1. Load your custom trained model weights
model = YOLO(r"runs\detect\pest_detection_model-6\weights\best.pt")

# 2. Export the model to ONNX format
# Using imgsz=640 to match your training, and simplify=True to remove redundant nodes
success = model.export(
    format="onnx", 
    imgsz=640, 
    simplify=True, 
    opset=12 # Standard opset version recommended for TVM and edge compilation
)
print("Export successful:", success)