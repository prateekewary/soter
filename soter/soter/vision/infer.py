import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .. import config

_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_segformer(checkpoint_dir=None):
    checkpoint_dir = checkpoint_dir or config.SEGFORMER_FINAL_DIR
    processor = SegformerImageProcessor.from_pretrained(checkpoint_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir).to(_device)
    model.eval()
    return processor, model


def predict_damage_mask(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
