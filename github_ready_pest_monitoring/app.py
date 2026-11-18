from collections import defaultdict
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from inference import generate_gradcam, predict_image


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "pest_detection_model"
    / "weights"
    / "best.pt"
)
AVAILABLE_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "pest_detection_model-6"
    / "weights"
    / "best.pt"
)
MODEL_OPTIONS = {
    "PyTorch (.pt)": MODEL_PATH,
    "ONNX (.onnx)": MODEL_PATH.with_suffix(".onnx"),
}
AVAILABLE_MODEL_OPTIONS = {
    "PyTorch (.pt)": AVAILABLE_MODEL_PATH,
    "ONNX (.onnx)": AVAILABLE_MODEL_PATH.with_suffix(".onnx"),
}

ADVISORIES = {
    "fall-armyworm-egg": (
        "Fall armyworm eggs may be on the leaves. Check under nearby leaves "
        "for small caterpillars."
    ),
    "fall-armyworm-frass": (
        "Small droppings may mean caterpillars are inside the maize plant. "
        "Check inside the whorl and nearby plants."
    ),
    "fall-armyworm-larva": (
        "Fall armyworm caterpillars may be present. Check nearby plants and "
        "ask your local farm adviser what to do."
    ),
    "fall-armyworm-larval-damage": (
        "Caterpillar damage may be present. Check inside the maize plant "
        "and look for caterpillars on nearby plants."
    ),
    "healthy-maize": (
        "This plant looks healthy in the photo. Keep checking your other plants."
    ),
    "maize-streak-disease": (
        "The leaves may have maize streak disease. Ask your local farm adviser "
        "to check the plant."
    ),
}


@st.cache_resource
def load_model(checkpoint: str) -> YOLO:
    return YOLO(checkpoint)


def build_advisory(detections: dict[str, list[float]]) -> str:
    if not detections:
        return (
            "I did not find any of the listed pests or diseases in this photo. "
            "Check your other plants too."
        )

    messages = []
    for label, confidences in sorted(
        detections.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        guidance = ADVISORIES.get(
            label,
            "Please ask your local farm adviser to check this plant.",
        )
        class_name = label.replace("-", " ")
        messages.append(
            f"{class_name.capitalize()} seen ({len(confidences)}). {guidance}"
        )

    return (
        "Message for the farmer: "
        + " ".join(messages)
        + " This is only a photo check. Please look at the plants in the field."
    )


st.set_page_config(page_title="Maize Pest Detection", page_icon="🌱", layout="wide")
st.title("Maize Pest Detection")
st.write(
    "Upload a crop image to run your trained YOLOv8 model and view detections."
)

checkpoint_options = {
    label: primary_path if primary_path.is_file() else fallback_path
    for label, primary_path in MODEL_OPTIONS.items()
    for fallback_path in (AVAILABLE_MODEL_OPTIONS[label],)
    if primary_path.is_file() or fallback_path.is_file()
}

if not checkpoint_options:
    st.error(
        "Model weights were not found. Expected "
        "`runs/detect/pest_detection_model/weights/best.pt` or "
        "`runs/detect/pest_detection_model-6/weights/best.pt`. "
        "Place the trained checkpoint in one of those locations and restart the app."
    )
    st.stop()

with st.sidebar:
    st.header("Inference settings")
    model_format = st.selectbox("Model format", options=list(checkpoint_options))
    checkpoint_path = checkpoint_options[model_format]
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=0.95,
        value=0.10,
        step=0.01,
    )
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

is_onnx = model_format == "ONNX (.onnx)"
device = "cpu" if is_onnx or not torch.cuda.is_available() else 0
device_name = (
    "CPU (ONNX Runtime)"
    if is_onnx
    else (
        torch.cuda.get_device_name(0)
        if device == 0
        else "CPU"
    )
)

try:
    model = load_model(str(checkpoint_path))
except Exception as error:
    st.error(f"Could not load model checkpoint `{checkpoint_path}`: {error}")
    st.stop()

st.caption(f"Checkpoint: `{checkpoint_path.relative_to(Path(__file__).resolve().parent)}`")
with st.sidebar:
    st.caption(f"Device: {device_name}")

uploaded_file = st.file_uploader(
    "Upload a crop image",
    type=("jpg", "jpeg", "png", "webp"),
    help="Supported formats: JPG, PNG, and WebP.",
)

if uploaded_file is None:
    st.info("Choose an image to start detection.")
    st.stop()

try:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
except (UnidentifiedImageError, OSError) as error:
    st.error(f"Could not read the uploaded image: {error}")
    st.stop()

try:
    result = predict_image(model, image, confidence_threshold, device)
except Exception as error:
    st.error(f"Inference failed: {error}")
    st.stop()

annotated_bgr = result.plot(boxes=True, labels=True, conf=True)
annotated_image = Image.fromarray(annotated_bgr[:, :, ::-1].copy())

st.subheader("Detection result")
st.image(annotated_image, caption="Predictions with bounding boxes and confidence scores")

detections: defaultdict[str, list[float]] = defaultdict(list)
if result.boxes is not None:
    for class_id, confidence in zip(
        result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        label = str(result.names[int(class_id)])
        detections[label].append(float(confidence))

if not detections:
    st.info(
        "No boxes were found at this confidence setting. Try lowering the "
        "confidence threshold in the sidebar, or use a clearer, closer photo."
    )

if show_gradcam:
    st.subheader("Grad-CAM explanation")
    if result.boxes is None or len(result.boxes) == 0:
        st.info("A Grad-CAM explanation needs at least one detected object.")
    else:
        best_detection = int(result.boxes.conf.argmax().item())
        target_class_id = int(result.boxes.cls[best_detection].item())
        target_class_name = str(result.names[target_class_id])
        gradcam_checkpoint = checkpoint_path.with_suffix(".pt")

        if not gradcam_checkpoint.is_file():
            st.info(
                "Grad-CAM needs the matching PyTorch (.pt) weights. "
                f"Expected `{gradcam_checkpoint.relative_to(Path(__file__).resolve().parent)}`."
            )
        else:
            try:
                gradcam_model = (
                    model
                    if not is_onnx
                    else load_model(str(gradcam_checkpoint))
                )
                with st.spinner("Generating Grad-CAM explanation..."):
                    gradcam_result = generate_gradcam(
                        gradcam_model,
                        image,
                        target_class_id,
                        device,
                    )
                st.caption(
                    "Heatmap for the highest-confidence detection: "
                    f"{target_class_name.replace('-', ' ')} "
                    f"({float(result.boxes.conf[best_detection]):.0%}). "
                    "Highlighted areas show image regions that influenced "
                    "the selected class score; this is an explanation, not "
                    "a separate detection."
                )
                st.image(gradcam_result, caption="Grad-CAM heatmap overlay")
            except Exception as error:
                st.error(f"Could not generate the Grad-CAM explanation: {error}")

speed = result.speed
inference_ms = speed.get("inference", 0.0)
preprocess_ms = speed.get("preprocess", 0.0)
postprocess_ms = speed.get("postprocess", 0.0)

st.subheader("Telemetry")
metric_columns = st.columns(4)
metric_columns[0].metric("Inference", f"{inference_ms:.1f} ms")
metric_columns[1].metric("Device", device_name)
metric_columns[2].metric("Detections", sum(map(len, detections.values())))
metric_columns[3].metric(
    "Pre / post-process", f"{preprocess_ms:.1f} / {postprocess_ms:.1f} ms"
)
st.caption("Inference time is measured per image and varies with hardware and image size.")

st.subheader("Advisory Agent")
st.caption("A simple sample text message based on what the photo shows.")
st.text_area(
    "Message for the farmer",
    value=build_advisory(dict(detections)),
    height=150,
    disabled=True,
    label_visibility="collapsed",
)
