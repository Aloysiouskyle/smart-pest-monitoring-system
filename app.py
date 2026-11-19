from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

import streamlit as st
from PIL import Image

from advisory_agent import build_advisory
from forecasting_agent import RiskRules, assess_risk
from microclimate_sim import generate_telemetry


ROOT = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

FARMER_VIEW = "Farmer View"
RESEARCH_VIEW = "Developer/Research View"
SCENARIOS = {"Normal maize field": "normal", "Higher-risk scenario": "higher_risk"}
FARMER_LABELS = {
    "fall-armyworm-egg": "Possible Fall Armyworm eggs",
    "fall-armyworm-frass": "Possible Fall Armyworm droppings",
    "fall-armyworm-larva": "Possible Fall Armyworm caterpillars",
    "fall-armyworm-larval-damage": "Possible Fall Armyworm leaf damage",
    "healthy-maize": "Maize that looks healthy in this photo",
    "maize-streak-disease": "Possible maize streak disease",
}


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


def render_gradcam(result, image, checkpoint, device, *, research=True):
    if research:
        st.subheader("Grad-CAM explanation")
    if result.boxes is None or len(result.boxes) == 0:
        st.info("A Grad-CAM explanation needs at least one detected object.")
        return
    weights = checkpoint.with_suffix(".pt")
    if not weights.is_file():
        st.info(
            f"Grad-CAM requires the matching PyTorch weights at {weights.relative_to(ROOT)}."
            if research else "The explanation heatmap is unavailable for this photo check."
        )
        return
    try:
        from inference import generate_gradcam

        best = int(result.boxes.conf.argmax().item())
        class_id = int(result.boxes.cls[best].item())
        with st.spinner("Generating Grad-CAM explanation..."):
            overlay = generate_gradcam(session_model(weights, "gradcam_model"), image, class_id, device)
        st.image(overlay, caption="Grad-CAM heatmap overlay")
        if research:
            st.caption(f"Class heatmap for {result.names[class_id]} "
                       f"(highest detection confidence {float(result.boxes.conf[best]):.0%}).")
    except Exception as error:
        st.error(f"Could not generate the Grad-CAM explanation: {error}"
                 if research else "The explanation heatmap could not be created.")


@dataclass
class PhotoResult:
    image: Image.Image
    checkpoint: Path
    device: int | str
    device_name: str
    result: "Results"
    detections: dict[str, list[float]]


def check_photo(photo, checkpoint, confidence) -> PhotoResult:
    cache_key = (photo, checkpoint, confidence)
    cached = st.session_state.get("photo_result")
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    st.session_state.pop("photo_result", None)

    import torch
    from inference import predict_image

    image = Image.open(BytesIO(photo)).convert("RGB")
    onnx = checkpoint.suffix == ".onnx"
    device = "cpu" if onnx or not torch.cuda.is_available() else 0
    device_name = "CPU (ONNX Runtime)" if onnx else (
        torch.cuda.get_device_name(0) if device == 0 else "CPU"
    )
    model = session_model(checkpoint, "detection_model")
    result = predict_image(model, image, confidence, device)
    detections = defaultdict(list)
    if result.boxes is not None:
        for class_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            detections[str(result.names[int(class_id)])].append(float(score))
    checked = PhotoResult(image, checkpoint, device, device_name, result, dict(detections))
    st.session_state["photo_result"] = (cache_key, checked)
    return checked


def render_annotated_photo(photo, *, research=True):
    try:
        annotated = photo.result.plot(boxes=True, labels=True, conf=True)
        st.image(
            annotated[:, :, ::-1].copy(),
            caption="Maize photo with boxes marking possible findings",
        )
    except Exception as error:
        st.warning(f"Could not display the marked photo: {error}"
                   if research else "The marked photo could not be displayed.")


def render_farmer_photo(photo, photo_error):
    if photo_error:
        st.warning("The photo check is unavailable. Try uploading a clear maize photo again.")
    elif photo is None:
        st.caption("No photo submitted. You can still see the demonstration.")
    elif not photo.detections:
        st.info("The photo check found no signs it recognises in this image.")
    else:
        st.markdown("\n".join(
            f"- {FARMER_LABELS.get(label, 'Possible sign on maize')}"
            for label in sorted(photo.detections)
        ))
        st.caption("These are possible signs from one photo.")

    if photo is not None:
        with st.expander("Marked photo and explanation heatmap"):
            render_annotated_photo(photo, research=False)
            if st.checkbox("Show explanation heatmap", key="farmer_heatmap"):
                st.caption("The heatmap highlights areas used by the model.")
                render_gradcam(photo.result, photo.image, photo.checkpoint, photo.device, research=False)


def render_research_photo(photo, photo_error, explain):
    if photo_error:
        st.error(photo_error)
    if photo is None:
        return

    st.header("Detection result")
    render_annotated_photo(photo)
    if not photo.detections:
        st.info("No boxes at this confidence setting. Try a lower threshold or a clearer photo.")
    elif photo.result.boxes is not None:
        boxes = photo.result.boxes
        st.dataframe([
            {"Class": photo.result.names[int(class_id)], "Confidence": score,
             "x1": bounds[0], "y1": bounds[1], "x2": bounds[2], "y2": bounds[3]}
            for class_id, score, bounds in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            )
        ])
    if explain:
        render_gradcam(photo.result, photo.image, photo.checkpoint, photo.device)

    speed = photo.result.speed
    columns = st.columns(4)
    columns[0].metric("Inference", f"{speed.get('inference', 0):.1f} ms")
    columns[1].metric("Device", photo.device_name)
    columns[2].metric("Detections", sum(map(len, photo.detections.values())))
    columns[3].metric(
        "Pre / post-process",
        f"{speed.get('preprocess', 0):.1f} / {speed.get('postprocess', 0):.1f} ms",
    )
    with st.expander("Photo advisory"):
        st.write(photo_advisory(photo.detections))


def render_weather(rows, scenario, research):
    st.caption(f"{scenario} · 72 hours of simulated readings.")
    if research:
        st.caption(f"{rows[0]['timestamp']} to {rows[-1]['timestamp']}")
    columns = st.columns(2)
    columns[0].metric("Average temperature", f"{mean(r['temperature_c'] for r in rows):.1f} °C")
    columns[1].metric("Average humidity", f"{mean(r['relative_humidity_pct'] for r in rows):.1f}%")
    columns[0].metric("Total rainfall", f"{sum(r['rainfall_mm_per_hour'] for r in rows):.1f} mm")
    columns[1].metric("Average soil moisture", f"{mean(r['soil_moisture_pct'] for r in rows):.1f}%")
    if research:
        st.dataframe(rows)


def render_risk(rows, photo, image_unprocessed, research, language):
    try:
        assessment = assess_risk(
            rows, None if photo is None else photo.detections,
            image_unprocessed=image_unprocessed,
        )
    except ValueError as error:
        st.error(f"Assessment could not use the supplied evidence: {error}")
        return

    st.metric("Simulated Fall Armyworm risk", assessment.risk_level)
    if research:
        for factor in assessment.contributing_factors:
            st.write(factor)
        with st.expander("Risk rules"):
            rules = RiskRules()
            st.write(
                f"One climate point requires at least {rules.min_matching_hours} of "
                f"{assessment.telemetry_hours} hours with both temperature ≥ "
                f"{rules.min_temperature_c:g} °C and relative humidity ≥ "
                f"{rules.min_humidity_pct:g}%."
            )
            st.write(
                f"One photo point requires a Fall Armyworm box with confidence ≥ "
                f"{rules.min_detection_confidence:.0%}."
            )
            st.write("0 points = Low · 1 point = Moderate · 2 points = High")
        st.caption(assessment.limitation)
        st.subheader("Farmer SMS previews")
    else:
        st.caption("Demonstration using simulated weather. The risk estimate is unvalidated and cannot forecast your field.")
        st.subheader("Recommended action")

    try:
        advisory = build_advisory(assessment)
    except ValueError as error:
        st.warning(f"SMS preview unavailable: {error}")
        return
    messages = {"English": advisory.english, "Draft Luganda": advisory.luganda}
    for name in messages if research else (language,):
        st.markdown(f"**{name}**")
        st.write(messages[name])
        if name == "Draft Luganda" and advisory.luganda_requires_review:
            st.caption("Draft translation awaiting human review before use with farmers.")
        if research:
            st.caption(f"{len(messages[name])}/160 characters · GSM-7")


st.set_page_config(page_title="Maize pest monitoring", page_icon="🌱", layout="centered")
st.title("Maize pest monitoring")
view = st.selectbox("View", (FARMER_VIEW, RESEARCH_VIEW), key="view")
research = view == RESEARCH_VIEW
language = "English"
scenario, seed = "Normal maize field", 42
options = available_checkpoints()
checkpoint = next(iter(options.values()), None)
confidence, explain = 0.10, True
if research:
    with st.sidebar:
        st.header("Inference settings")
        if options:
            checkpoint = options[st.selectbox("Model format", list(options), key="backend")]
        else:
            st.caption("Detection weights are missing.")
        confidence = st.slider("Confidence threshold", 0.01, 0.95, 0.10, 0.01, key="confidence")
        explain = st.checkbox("Show Grad-CAM explanation", value=True, key="gradcam")
        st.header("Simulation settings")
        scenario = st.selectbox("Scenario", list(SCENARIOS), key="scenario")
        seed = st.number_input("Random seed", value=42, step=1, key="seed")
else:
    language = st.selectbox("Advisory language", ("English", "Draft Luganda"), key="language")

risk_panel = st.container(border=True)
st.subheader("Maize photo check")
upload = st.file_uploader("Upload a maize photo (optional)", type=("jpg", "jpeg", "png", "webp"), key="photo")
photo, photo_error = None, None
if upload is None or checkpoint is None:
    st.session_state.pop("photo_result", None)
if upload is not None:
    if checkpoint is None:
        photo_error = "Detection weights are missing. Place best.pt or best.onnx in a configured weights folder."
    else:
        try:
            with st.spinner("Checking your maize photo…"):
                photo = check_photo(upload.getvalue(), checkpoint, confidence)
        except Exception as error:
            photo_error = f"Photo processing failed: {error}"

rows = generate_telemetry(SCENARIOS[scenario], seed=seed)
with risk_panel:
    render_risk(rows, photo, upload is not None and photo is None, research, language)
if research:
    if checkpoint is not None:
        st.caption(f"Checkpoint: {checkpoint.relative_to(ROOT)}")
    render_research_photo(photo, photo_error, explain)
    st.header("Simulated microclimate")
    render_weather(rows, scenario, research)
else:
    render_farmer_photo(photo, photo_error)
    with st.expander("Simulated weather readings"):
        render_weather(rows, scenario, research)
