import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class YOLOClassTarget:
    def __init__(self, class_id: int) -> None:
        self.class_id = class_id

    def __call__(
        self,
        model_output: torch.Tensor | tuple[torch.Tensor, object],
    ) -> torch.Tensor:
        predictions = (
            model_output[0] if isinstance(model_output, tuple) else model_output
        )
        if predictions.ndim == 3:
            predictions = predictions[0]
        class_scores = predictions[4 + self.class_id, :]
        return class_scores.max()


def predict_image(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    device: int | str,
) -> Results:
    return model.predict(
        source=image,
        conf=confidence,
        device=device,
        verbose=False,
    )[0]


def generate_gradcam(
    model: YOLO,
    image: Image.Image,
    class_id: int,
    device: int | str,
) -> Image.Image:
    target_model = model.model
    target_device = torch.device(
        f"cuda:{device}" if isinstance(device, int) else device
    )
    target_model.to(target_device)
    target_model.eval()

    resized = image.convert("RGB")
    resized.thumbnail((640, 640), Image.Resampling.BILINEAR)
    letterboxed = Image.new("RGB", (640, 640), (114, 114, 114))
    offset = ((640 - resized.width) // 2, (640 - resized.height) // 2)
    letterboxed.paste(resized, offset)

    input_tensor = (
        torch.from_numpy(np.asarray(letterboxed, dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(target_device)
        .requires_grad_(True)
    )
    rgb_image = np.asarray(letterboxed, dtype=np.float32) / 255.0
    target_layer = target_model.model[-2]

    with GradCAM(model=target_model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[YOLOClassTarget(class_id)],
        )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )
    return Image.fromarray(visualization)