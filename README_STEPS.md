# Mudra CNN Project Files

Classes:
- pataka
- tripataka
- katakamukham

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare data from Hugging Face

```bash
python 00_download_prepare_data.py --source hf --out mudra_dataset
```

If one class has 0 images, inspect class names:

```bash
python 00_download_prepare_data.py --source hf --list-classes
```

## 3. Train

```bash
python 01_train_resnet18.py --data mudra_dataset --epochs 15 --batch-size 32
```

## 4. Evaluate on online test set

```bash
python 02_evaluate_mudra_classifier.py --data mudra_dataset --split test --model models/mudra_resnet18_best.pth
```

## 5. Evaluate on your Jetson webcam test images

Put your images in:

```text
real_test/
  pataka/
  tripataka/
  katakamukham/
```

Then run:

```bash
python 02_evaluate_mudra_classifier.py --data real_test --split . --model models/mudra_resnet18_best.pth
```

## 6. Copy to Jetson

Copy:
- models/mudra_resnet18_best.pth
- 03_live_mudra_demo.py

Then run:

```bash
python3 03_live_mudra_demo.py --model models/mudra_resnet18_best.pth --camera 0
```
