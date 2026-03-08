# Dog-Cat Identifier

A machine learning project for classifying images and performing real-time detection of cats and dogs using TensorFlow/Keras.
https://drive.google.com/drive/folders/15Lvq16FyFgKlYZi61byo3pEN4XY6H7tm?usp=sharing

## Quick Start

### 1. Setup Environment

```powershell
python -m venv dogcat-env
.\dogcat-env\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow numpy matplotlib pillow opencv-python scikit-learn
```

### 2. Prepare Dataset

Place your dataset in `dogcat-env/dataset/` with this structure:

```
dataset/
├── train/
│   ├── dogs/
│   └── cats/
└── validation/
    ├── dogs/
    └── cats/
```

Or download from Kaggle:
```powershell
python src/download_dataset.py
```

### 3. Train Model

```powershell
python src/train_85percent.py
```

### 4. Test Model

```powershell
python src/test_85.py
```

### 5. Real-Time Detection

```powershell
python src/simple_detector.py
```

Press 'q' to quit.

## Available Scripts

| Script | Purpose |
|--------|---------|
| `train_85percent.py` | Train optimized 85% accuracy model |
| `test_85.py` | Test model on static images |
| `simple_detector.py` | Real-time webcam detection |
| `download_dataset.py` | Download Dogs vs Cats dataset |
| `balance_dataset.py` | Balance imbalanced dataset |
| `chart.py` | Generate performance visualizations |

## Pre-trained Models

- `best_model_85.keras` - 85% accuracy checkpoint
- `cat_dog_classifier_85percent.keras` - Production model
- `cat_dog_classifier.h5` - Legacy format model

## Requirements

- Python 3.10+
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy, Matplotlib, Pillow
- Webcam (optional, for real-time detection)


## Features

✅ Train custom CNN models  
✅ Test on static images  
✅ Real-time webcam detection  
✅ GUI interface for detection  
✅ Dataset management tools  
✅ Performance visualization  
✅ 85% accuracy pre-trained model  

## Troubleshooting

**Camera not opening?**
- Check camera permissions
- Ensure no other app is using the camera
- Try `camera_detector.py` instead

**Low accuracy?**
- Verify dataset structure
- Check image quality and format
- Ensure proper normalization
- Train with more epochs

**Missing model?**
- Download pre-trained model from releases
- Or train a new one with `train_85percent.py`

## License

Educational project


