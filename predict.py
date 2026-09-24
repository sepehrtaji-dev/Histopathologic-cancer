#!/usr/bin/env python3
"""
Nova CLI — Terminal-based histopathology cancer predictor
Usage:
    python predict.py image.tif
    python predict.py path/to/images/          # batch mode
    python predict.py                          # interactive mode
"""

import sys
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import CancerModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "checkpoints/best_model.pt"
IMAGE_SIZE   = 96
THRESHOLD    = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED    = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "[91m"
GREEN  = "[92m"
YELLOW = "[93m"
BLUE   = "[94m"
CYAN   = "[96m"
BOLD   = "[1m"
DIM    = "[2m"
RESET  = "[0m"

def clr(text, *codes):
    return "".join(codes) + text + RESET


# ── UI helpers ────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN}{BOLD}  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ {RESET}
{CYAN}{BOLD}  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗{RESET}
{CYAN}{BOLD}  ██╔██╗ ██║██║   ██║██║   ██║███████║{RESET}
{CYAN}{BOLD}  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║{RESET}
{CYAN}{BOLD}  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║{RESET}
{CYAN}{BOLD}  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝{RESET}
{DIM}  Histopathology Cancer Detector  •  CLI{RESET}
"""

def bar(probability: float, width: int = 40) -> str:
    filled = int(probability * width)
    empty  = width - filled
    color  = RED if probability >= THRESHOLD else GREEN
    return color + "█" * filled + RESET + DIM + "░" * empty + RESET


def spinner(message: str, seconds: float = 0.8):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end    = time.time() + seconds
    i      = 0
    while time.time() < end:
        print(f"  {CYAN}{frames[i % len(frames)]}{RESET}  {message}", end="", flush=True)
        time.sleep(0.08)
        i += 1
    print("" + " " * (len(message) + 10) + "", end="")


def divider(char="─", width=52):
    print(f"  {DIM}{char * width}{RESET}")


def print_result(path: str, probability: float):
    cancer    = probability >= THRESHOLD
    label     = clr("● CANCER DETECTED", RED, BOLD) if cancer else clr("● Non-cancer", GREEN, BOLD)
    pct       = f"{probability * 100:.2f}%"
    pct_color = clr(pct, RED if cancer else GREEN, BOLD)

    print()
    divider()
    print(f"  {BOLD}File   {RESET}  {DIM}{Path(path).name}{RESET}")
    print(f"  {BOLD}Result {RESET}  {label}")
    print(f"  {BOLD}Prob   {RESET}  {pct_color}  {bar(probability)}")
    divider()

    if cancer:
        print(f"
  {YELLOW}⚠  For research / educational use only.{RESET}")
        print(f"  {YELLOW}   This is not a medical diagnosis.{RESET}")
    print()


def print_batch_summary(results: list):
    total    = len(results)
    detected = sum(1 for _, p in results if p >= THRESHOLD)
    print()
    divider("═")
    print(f"  {BOLD}Batch Summary{RESET}")
    divider()
    print(f"  Total images  : {BOLD}{total}{RESET}")
    print(f"  Cancer        : {clr(str(detected), RED, BOLD)}")
    print(f"  Non-cancer    : {clr(str(total - detected), GREEN, BOLD)}")
    divider("═")
    print()


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    if not Path(MODEL_PATH).exists():
        print(clr(f"
  ✗  Checkpoint not found: {MODEL_PATH}", RED, BOLD))
        print(f"  {DIM}Run python train.py first.{RESET}
")
        sys.exit(1)

    spinner("Loading Nova model…")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    model = CancerModel(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_auc = checkpoint.get("val_auc")
    auc_str = f"  Val AUC : {BOLD}{val_auc:.4f}{RESET}" if val_auc else ""

    print(f"  {GREEN}✔{RESET}  Model loaded")
    print(f"  Device  : {BOLD}{DEVICE}{RESET}")
    if auc_str:
        print(auc_str)
    print()

    return model, transform


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_image(path: str, model, transform) -> float:
    image  = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    return prob


def run_single(path: str, model, transform):
    if not Path(path).exists():
        print(clr(f"
  ✗  File not found: {path}", RED))
        sys.exit(1)
    spinner(f"Analysing {Path(path).name}…")
    prob = predict_image(path, model, transform)
    print_result(path, prob)


def run_batch(folder: str, model, transform):
    folder = Path(folder)
    images = sorted([f for f in folder.iterdir()
                     if f.suffix.lower() in SUPPORTED])

    if not images:
        print(clr(f"
  ✗  No supported images found in {folder}", RED))
        sys.exit(1)

    print(f"  {BOLD}Found {len(images)} image(s) in {folder}{RESET}
")
    results = []

    for i, img_path in enumerate(images, 1):
        spinner(f"[{i}/{len(images)}]  {img_path.name}")
        prob = predict_image(str(img_path), model, transform)
        results.append((str(img_path), prob))

        cancer = prob >= THRESHOLD
        label  = clr("CANCER", RED, BOLD) if cancer else clr("Normal", GREEN)
        pct    = f"{prob * 100:.1f}%"
        print(f"  {'●' if cancer else '○'}  {img_path.name:<35} {label:<20} {pct}")

    print_batch_summary(results)


def run_interactive(model, transform):
    print(f"  {DIM}Type an image path, a folder, or {BOLD}quit{RESET}{DIM} to exit.{RESET}
")

    while True:
        try:
            raw = input(f"  {CYAN}nova>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"

  {DIM}Goodbye.{RESET}
")
            break

        if not raw:
            continue

        if raw.lower() in {"quit", "exit", "q"}:
            print(f"
  {DIM}Goodbye.{RESET}
")
            break

        path = Path(raw)

        if path.is_dir():
            run_batch(str(path), model, transform)
        elif path.is_file() and path.suffix.lower() in SUPPORTED:
            spinner(f"Analysing {path.name}…")
            prob = predict_image(str(path), model, transform)
            print_result(str(path), prob)
        else:
            print(clr(f"
  ✗  Not a valid image or folder: {raw}
", YELLOW))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print(BANNER)

    model, transform = load_model()

    args = sys.argv[1:]

    if not args:
        run_interactive(model, transform)

    elif len(args) == 1:
        target = Path(args[0])
        if target.is_dir():
            run_batch(str(target), model, transform)
        else:
            run_single(str(target), model, transform)

    else:
        # Multiple files passed
        results = []
        for arg in args:
            path = Path(arg)
            if not path.exists():
                print(clr(f"  ✗  Skipping (not found): {arg}", YELLOW))
                continue
            spinner(f"Analysing {path.name}…")
            prob = predict_image(str(path), model, transform)
            results.append((str(path), prob))
            cancer = prob >= THRESHOLD
            label  = clr("CANCER", RED, BOLD) if cancer else clr("Normal", GREEN)
            print(f"  {'●' if cancer else '○'}  {path.name:<35} {label}  {prob*100:.1f}%")

        if results:
            print_batch_summary(results)


if __name__ == "__main__":
    main()
