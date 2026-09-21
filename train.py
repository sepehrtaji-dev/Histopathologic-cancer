
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import HistopathDataset
from model import CancerModel


DATA_DIR = "data"
CSV_FILE = "data/_labels.csv"

IMAGE_SIZE = 96
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_SIZE = 0.2
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate(model, loader):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(images)
            probs = torch.sigmoid(logits).flatten()

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    probabilities = np.array(probabilities)
    labels = np.array(labels)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    auc = roc_auc_score(
        labels,
        probabilities
    )

    return accuracy, auc


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

        print("CUDA available")
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:
        print("CUDA not available. Using CPU.")

    print(f"Device: {DEVICE}")

    dataset = HistopathDataset(
        data_dir=DATA_DIR,
        csv_file=CSV_FILE,
        image_size=IMAGE_SIZE,
        training=True
    )

    labels = dataset.df[
        dataset.label_col
    ].astype(int).values

    indices = np.arange(len(dataset))

    train_indices, validation_indices = (
        train_test_split(
            indices,
            test_size=VALIDATION_SIZE,
            random_state=SEED,
            stratify=labels
        )
    )

    train_dataset = Subset(
        dataset,
        train_indices
    )

    validation_base = HistopathDataset(
        data_dir=DATA_DIR,
        csv_file=CSV_FILE,
        image_size=IMAGE_SIZE,
        training=False
    )

    validation_dataset = Subset(
        validation_base,
        validation_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    model = CancerModel(
        pretrained=True
    ).to(DEVICE)

    train_labels = labels[train_indices]

    positive = np.sum(train_labels == 1)
    negative = np.sum(train_labels == 0)

    pos_weight = torch.tensor(
        [negative / max(positive, 1)],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    Path("checkpoints").mkdir(
        exist_ok=True
    )

    best_auc = 0.0
    patience = 4
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for images, batch_labels in train_loader:
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            batch_labels = batch_labels.to(
                DEVICE,
                non_blocking=True
            ).unsqueeze(1)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):
                logits = model(images)

                loss = criterion(
                    logits,
                    batch_labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        validation_accuracy, validation_auc = (
            evaluate(
                model,
                validation_loader
            )
        )

        scheduler.step(validation_auc)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"Val AUC: {validation_auc:.4f}"
        )

        if validation_auc > best_auc:
            best_auc = validation_auc
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "image_size":
                        IMAGE_SIZE,
                    "val_auc":
                        validation_auc
                },
                "checkpoints/best_model.pt"
            )

            print(
                f"  Best model saved "
                f"(AUC: {validation_auc:.4f})"
            )

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping.")
                break

    print()
    print("Training finished.")
    print(
        f"Best validation AUC: "
        f"{best_auc:.4f}"
    )


if __name__ == "__main__":
    main()

