"""
Run live webcam mudra classification.

Usage on Jetson:
  python3 03_live_mudra_demo.py --model models_v2_9class/mudra_resnet18_best.pth --camera 0

Press q to quit.
"""

import argparse
import time

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models_v2_9class/mudra_resnet18_best.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    ckpt = torch.load(args.model, map_location=device)
    class_names = ckpt["class_names"]

    model = build_model(len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break

        # Convert OpenCV BGR frame to PIL RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        x = tfm(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        label = class_names[pred_idx.item()]
        conf_val = conf.item()

        if conf_val < args.confidence_threshold:
            display_text = f"Unknown ({conf_val:.2f})"
        else:
            display_text = f"{label} ({conf_val:.2f})"

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(frame, display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Live Mudra Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
