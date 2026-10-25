import os

import numpy as np
import torch
import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

import config

_device = "cuda" if torch.cuda.is_available() else "cpu"
_segformer_processor = None
_segformer_model = None
_unet_model = None

_unet_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_segformer():
    global _segformer_processor, _segformer_model
    if _segformer_model is not None:
        return _segformer_processor, _segformer_model
    if not os.path.isdir(config.SEGFORMER_DIR):
        return None, None
    _segformer_processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_DIR)
    _segformer_model = SegformerForSemanticSegmentation.from_pretrained(config.SEGFORMER_DIR).to(_device)
    _segformer_model.eval()
    return _segformer_processor, _segformer_model


def load_unet():
    global _unet_model
    if _unet_model is not None:
        return _unet_model
    if not os.path.exists(config.UNET_PATH):
        return None
    import segmentation_models_pytorch as smp

    _unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=config.NUM_LABELS if hasattr(config, "NUM_LABELS") else 4).to(_device)
    _unet_model.load_state_dict(torch.load(config.UNET_PATH, map_location=_device))
    _unet_model.eval()
    return _unet_model


def list_sample_chips():
    if not os.path.isdir(config.SAMPLE_CHIP_DIR):
        return []
    return sorted(f for f in os.listdir(config.SAMPLE_CHIP_DIR) if f.lower().endswith(".png"))


def checkpoint_status():
    return {
        "segformer": os.path.isdir(config.SEGFORMER_DIR),
        "unet": os.path.exists(config.UNET_PATH),
    }


def predict_damage_mask(image, model_name="SegFormer v2"):
    orig_size = image.size

    if model_name == "U-Net Baseline":
        model = load_unet()
        if model is None:
            raise RuntimeError(f"U-Net checkpoint not found at {config.UNET_PATH}")
        pixel_values = _unet_transform(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            logits = model(pixel_values)
    else:
        processor, model = load_segformer()
        if model is None:
            raise RuntimeError(f"SegFormer checkpoint not found at {config.SEGFORMER_DIR}")
        inputs = processor(image, return_tensors="pt").to(_device)
        with torch.no_grad():
            logits = model(**inputs).logits

    upsampled = torch.nn.functional.interpolate(
        logits, size=orig_size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)