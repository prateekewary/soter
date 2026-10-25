import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .. import config
from .dataset import XBDSegDataset, build_oversampled_paths

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def train(processor, epochs=6, batch_size=8, lr=1e-4):
    assert smp is not None, "pip install segmentation-models-pytorch first"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oversampled_paths = build_oversampled_paths(config.XBD_ROOT, split="train")
    train_ds = XBDSegDataset(config.XBD_ROOT, processor, split="train", image_paths=oversampled_paths)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    unet_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=config.NUM_LABELS,
    ).to(device)

    class_weights_tensor = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = CrossEntropyLoss(weight=class_weights_tensor, ignore_index=255)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=lr)

    unet_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = unet_model(pixel_values)
            upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(upsampled, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} avg loss {total_loss / len(train_loader):.4f}")

    torch.save(unet_model.state_dict(), config.UNET_PATH)
    return unet_model
