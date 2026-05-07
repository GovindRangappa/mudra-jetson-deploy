# Mudra CNN Project Files

Classes:
- pataka
- tripataka
- alapadma
- musti
- ardhapataka
- ardhachandra
- mayura
- shikhara
- kapittha

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare data from Hugging Face

```bash
python 00_download_prepare_data.py --source hf --out mudra_dataset --max-per-class 2500
```

If one class has 0 images, inspect class names:

```bash
python 00_download_prepare_data.py --source hf --list-classes
```

If you want only a subset of mudras, pass `--targets`:

```bash
python 00_download_prepare_data.py --source hf --out mudra_dataset --max-per-class 3000 --targets pataka tripataka alapadma musti ardhapataka
```

## 3. Train

```bash
python 01_train_resnet18.py --data mudra_dataset --epochs 15 --batch-size 32
```

Default output folder is now `models_v2_9class/` so this model can coexist with older versions.

## 4. Evaluate on online test set

```bash
python 02_evaluate_mudra_classifier.py --data mudra_dataset --split test --model models_v2_9class/mudra_resnet18_best.pth
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
python 02_evaluate_mudra_classifier.py --data real_test --split . --model models_v2_9class/mudra_resnet18_best.pth
```

## 6. Copy to Jetson

Copy:
- models_v2_9class/mudra_resnet18_best.pth
- 03_live_mudra_demo.py

Then run:

```bash
python3 03_live_mudra_demo.py --model models_v2_9class/mudra_resnet18_best.pth --camera 0
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
```
