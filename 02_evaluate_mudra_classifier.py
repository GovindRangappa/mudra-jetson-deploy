"""
Evaluate the trained classifier on a dataset split.

Usage:
  python 02_evaluate_mudra_classifier.py --data mudra_dataset --split test --model models/mudra_resnet18_best.pth

For your Jetson webcam pictures, place them like:
  real_test/
    pataka/
    tripataka/
    katakamukham/

Then run:
  python 02_evaluate_mudra_classifier.py --data real_test --split . --model models/mudra_resnet18_best.pth
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mudra_dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="models/mudra_resnet18_best.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    class_names = ckpt["class_names"]

    eval_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    data_path = Path(args.data) if args.split == "." else Path(args.data) / args.split
    ds = datasets.ImageFolder(data_path, transform=eval_tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(y.numpy().tolist())

    print("\nClass names from model:", class_names)
    print("Class names from folder:", ds.classes)

    print("\nClassification Report:")
    print(classification_report(
      all_true,
      all_pred,
      labels=ds.classes,
      target_names=ds.classes,
      zero_division=0,
      digits=4
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

if __name__ == "__main__":
    main()
