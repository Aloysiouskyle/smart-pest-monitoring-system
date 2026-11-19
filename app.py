from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean

import streamlit as st
from PIL import Image

from advisory_agent import build_advisory
from forecasting_agent import assess_risk
from microclimate_sim import generate_telemetry


ROOT = Path(__file__).resolve().parent


def available_checkpoints() -> dict[str, Path]:
    options = {}
    for label, filename in (("PyTorch (.pt)", "best.pt"), ("ONNX (.onnx)", "best.onnx")):
        for run in ("pest_detection_model", "pest_detection_model-6"):
            path = ROOT / "runs" / "detect" / run / "weights" / filename
            if path.is_file():
                options[label] = path
                break
    return options


def session_model(path: Path, role: str):
    """Keep prediction and gradient models private to this Streamlit session."""
    from ultralytics import YOLO

    cached = st.session_state.get(role)
    if cached is None or cached[0] != path:
        cached = (path, YOLO(str(path)))
        st.session_state[role] = cached
    return cached[1]


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


def photo_advisory(detections: dict[str, list[float]]) -> str:
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


def render_gradcam(result, image, checkpoint, device):
    st.subheader("Grad-CAM explanation")
    if result.boxes is None or len(result.boxes) == 0:
        st.info("A Grad-CAM explanation needs at least one detected object.")
        return
    weights = checkpoint.with_suffix(".pt")
    if not weights.is_file():
        st.info(f"Grad-CAM requires the matching PyTorch weights at {weights.relative_to(ROOT)}.")
        return
    try:
        from inference import generate_gradcam

        best = int(result.boxes.conf.argmax().item())
        class_id = int(result.boxes.cls[best].item())
        with st.spinner("Generating Grad-CAM explanation..."):
            overlay = generate_gradcam(session_model(weights, "gradcam_model"), image, class_id, device)
        st.image(overlay, caption="Grad-CAM heatmap overlay")
        st.caption(f"Class heatmap for {result.names[class_id]} "
                   f"(highest detection confidence {float(result.boxes.conf[best]):.0%}).")
    except Exception as error:
        st.error(f"Could not generate the Grad-CAM explanation: {error}")


def render_detection(upload) -> dict[str, list[float]] | None:
    options = available_checkpoints()
    if not options:
        st.warning("Detection weights are missing. Place best.pt or best.onnx in a configured weights folder.")
        return None

    with st.sidebar:
        st.header("Inference settings")
        checkpoint = options[st.selectbox("Model format", list(options))]
        confidence = st.slider("Confidence threshold", 0.01, 0.95, 0.10, 0.01)
        explain = st.checkbox("Show Grad-CAM explanation", value=True)
    st.caption(f"Checkpoint: {checkpoint.relative_to(ROOT)}")
    if upload is None:
        st.info("Choose an image to start detection.")
        return None

    try:
        import torch
        from inference import predict_image

        image = Image.open(BytesIO(upload.getvalue())).convert("RGB")
        onnx = checkpoint.suffix == ".onnx"
        device = "cpu" if onnx or not torch.cuda.is_available() else 0
        device_name = "CPU (ONNX Runtime)" if onnx else (
            torch.cuda.get_device_name(0) if device == 0 else "CPU"
        )
        model = session_model(checkpoint, "detection_model")
        result = predict_image(model, image, confidence, device)
        annotated = result.plot(boxes=True, labels=True, conf=True)
        detections = defaultdict(list)
        if result.boxes is not None:
            for class_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                detections[str(result.names[int(class_id)])].append(float(score))
    except Exception as error:
        st.error(f"Photo processing failed: {error}")
        return None

    st.subheader("Detection result")
    st.image(annotated[:, :, ::-1].copy(), caption="Predictions with bounding boxes and confidence scores")
    if not detections:
        st.info("No boxes at this confidence setting. Try a lower threshold or a clearer photo.")
    if explain:
        render_gradcam(result, image, checkpoint, device)

    st.subheader("Telemetry")
    speed = result.speed
    columns = st.columns(4)
    columns[0].metric("Inference", f"{speed.get('inference', 0):.1f} ms")
    columns[1].metric("Device", device_name)
    columns[2].metric("Detections", sum(map(len, detections.values())))
    columns[3].metric("Pre / post-process",
                      f"{speed.get('preprocess', 0):.1f} / {speed.get('postprocess', 0):.1f} ms")
    with st.expander("Photo advisory"):
        st.write(photo_advisory(detections))
    return dict(detections)


def render_assessment(detections, image_unprocessed):
    st.header("Simulated microclimate")
    scenarios = {"Normal maize field": "normal", "Higher-risk scenario": "higher_risk"}
    left, right = st.columns(2)
    scenario = left.selectbox("Scenario", list(scenarios), key="scenario")
    seed = right.number_input("Random seed", value=42, step=1, key="seed")
    rows = generate_telemetry(scenarios[scenario], seed=seed)
    st.caption(f"72 synthetic hourly readings from {rows[0]['timestamp']} to {rows[-1]['timestamp']}.")
    columns = st.columns(4)
    columns[0].metric("Mean temperature", f"{mean(r['temperature_c'] for r in rows):.1f} °C")
    columns[1].metric("Mean humidity", f"{mean(r['relative_humidity_pct'] for r in rows):.1f}%")
    columns[2].metric("Total rainfall", f"{sum(r['rainfall_mm_per_hour'] for r in rows):.1f} mm")
    columns[3].metric("Mean soil moisture", f"{mean(r['soil_moisture_pct'] for r in rows):.1f}%")

    try:
        assessment = assess_risk(rows, detections, image_unprocessed=image_unprocessed)
    except ValueError as error:
        st.error(f"Assessment could not use the supplied evidence: {error}")
        return
    st.metric("Simulated Fall Armyworm risk", assessment.risk_level)
    for factor in assessment.contributing_factors:
        st.write(factor)
    st.caption(assessment.limitation)

    st.subheader("Farmer SMS previews")
    try:
        advisory = build_advisory(assessment)
    except ValueError as error:
        st.warning(f"SMS preview unavailable: {error}")
        return
    for language, message in (("English", advisory.english), ("Draft Luganda", advisory.luganda)):
        st.markdown(f"**{language}**")
        st.write(message)
        st.caption(f"{len(message)}/160 characters · GSM-7")
    if advisory.luganda_requires_review:
        st.caption("The Luganda draft requires human review before use with farmers.")


st.set_page_config(page_title="Maize pest monitoring", page_icon="🌱", layout="wide")
st.title("Maize pest monitoring")
upload = st.file_uploader("Upload a crop image", type=("jpg", "jpeg", "png", "webp"))
detections = render_detection(upload)
render_assessment(detections, image_unprocessed=upload is not None and detections is None)
