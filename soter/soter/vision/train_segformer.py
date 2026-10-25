import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, Trainer, TrainingArguments

from .. import config
from .dataset import XBDSegDataset, build_oversampled_paths


class WeightedSegTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss_fct = CrossEntropyLoss(weight=self.class_weights, ignore_index=255)
        loss = loss_fct(upsampled, labels.long())
        return (loss, outputs) if return_outputs else loss


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def train(epochs=18, learning_rate=4e-5, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU found — switch the Colab runtime to GPU first."

    processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_CKPT, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.SEGFORMER_CKPT, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True
    ).to(device)

    oversampled_paths = build_oversampled_paths(config.XBD_ROOT, split="train")
    train_ds = XBDSegDataset(config.XBD_ROOT, processor, split="train", image_paths=oversampled_paths)
    val_ds = XBDSegDataset(config.XBD_ROOT, processor, split="test")

    class_weights_tensor = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)

    training_args = TrainingArguments(
        output_dir=config.SEGFORMER_OUT_DIR,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to=[],
        save_total_limit=2,
    )

    trainer = WeightedSegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        class_weights=class_weights_tensor,
    )
    trainer.train()

    model.save_pretrained(config.SEGFORMER_FINAL_DIR)
    processor.save_pretrained(config.SEGFORMER_FINAL_DIR)
    return model, processor


if __name__ == "__main__":
    train()
