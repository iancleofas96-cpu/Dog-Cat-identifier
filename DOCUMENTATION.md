# Dog-Cat Identifier Documentation

## Project Overview

**Dog-Cat Identifier** is a machine learning project that uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as either dogs or cats. The project includes multiple modules for training models, testing accuracy, and real-time detection using a webcam.

## Features

- **Model Training**: Build and train CNN models from scratch using custom datasets
- **Real-time Detection**: Performs live cat vs dog classification using a webcam
- **Multiple Interface Options**: GUI-based and command-line detection interfaces
- **Dataset Management**: Tools for downloading and balancing datasets
- **Performance Metrics**: Visualization of training progress and model accuracy
- **Pre-trained Models**: Includes 85% accuracy model for quick testing

## Project Structure

```
Dog-Cat-identifier/
├── DOCUMENTATION.md                    # This file
├── README.md                           # Quick start guide
├── best_model_85.keras                 # Best performing model checkpoint
├── cat_dog_classifier_85percent.keras  # 85% accuracy model (recommended)
├── cat_dog_classifier.h5               # Alternative model format (legacy)
│
├── src/                                # Source code directory
│   ├── train_85percent.py              # Train optimized 85% accuracy model
│   ├── test_85.py                      # Test 85% accuracy model
│   ├── simple_detector.py              # Real-time webcam detection
│   ├── download_dataset.py             # Download dataset from Kaggle
│   ├── balance_dataset.py              # Balance dataset for training
│   └── chart.py                        # Generate performance visualizations
│
├── dogcat-env/                         # Python virtual environment
│   ├── dataset/
│   │   ├── train/                      # Training images
│   │   │   ├── cats/
│   │   │   └── dogs/
│   │   └── validation/                 # Validation images
│   │       ├── cats/
│   │       └── dogs/
│   └── [other env files]
│
└── [Image files]                       # Generated metric charts
    ├── 1_metrics_comparison.png
    ├── 3_training_progress.png
    └── cat_dog_dashboard_2column.png
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Webcam (optional, for real-time detection)

### Step 1: Set Up Virtual Environment

```powershell
# Create virtual environment
python -m venv dogcat-env

# Activate it (Windows PowerShell)
.\dogcat-env\Scripts\Activate.ps1

# Or on Command Prompt
dogcat-env\Scripts\activate
```

### Step 2: Install Dependencies

```powershell
pip install --upgrade pip

pip install tensorflow numpy matplotlib pillow opencv-python scikit-learn
```

For the advanced GUI detector, also install:
```powershell
pip install pillow
```

### Step 3: Prepare Dataset

The project expects a dataset structure in `dogcat-env/dataset/`:

```
dataset/
├── train/
│   ├── dogs/          # Training dog images
│   └── cats/          # Training cat images
└── validation/
    ├── dogs/          # Validation dog images
    └── cats/          # Validation cat images
```

**Option A: Download from Kaggle**
```powershell
python src/download_dataset.py
```

**Option B: Manually arrange images**
- Download the "Dogs vs Cats" dataset from Kaggle
- Organize files according to the structure above

**Option C: Balance your dataset**
If your dataset is imbalanced:
```powershell
python src/balance_dataset.py
```

## Usage

### Recommended Execution Order

Follow these steps in sequence for the best workflow:

#### Step 1: Prepare Dataset
```powershell
# Option A: Download from Kaggle
python src/download_dataset.py

# Option B: Balance imbalanced dataset (if needed)
python src/balance_dataset.py
```

#### Step 2: Train Model
```powershell
# Train optimized model (85% accuracy target)
python src/train_85percent.py
```

This creates `best_model_85.keras` in the project root.

#### Step 3: Test Model
```powershell
# Test the trained model on static images
python src/test_85.py
```

#### Step 4: Real-Time Detection
```powershell
# Run real-time webcam detection
python src/simple_detector.py
```

Press 'q' to quit.

#### Step 5 (Optional): Generate Charts
```powershell
# Create performance visualization
python src/chart.py
```

Generates PNG files with metrics and training history.

---

### Alternative: Quick Start with Pre-trained Model

If you don't want to train, use the included 85% model directly:

```powershell
python src/simple_detector.py
```

The script automatically loads `cat_dog_classifier_85percent.keras`

## File Descriptions

### Core Scripts

#### `train_85percent.py`
Optimized training script achieving 85% validation accuracy.

**Key features:**
- MobileNetV2 pre-trained backbone for faster training
- Aggressive data augmentation with 224×224 images
- Early stopping and learning rate reduction
- Model checkpointing saves best weights
- Two-phase training for optimal convergence
- Batch size: 32, Epochs: ~35

**Output**: Creates `best_model_85.keras`

**Usage**:
```powershell
python src/train_85percent.py
```

#### `test_85.py`
Test script for evaluating the 85% accuracy model.

**Key features:**
- Load pre-trained model
- Test on static images from validation set
- Display predictions with confidence
- Shows processed images and predictions
- Comprehensive accuracy metrics

**Usage**:
```powershell
python src/test_85.py
```

#### `simple_detector.py`
Lightweight real-time detection using webcam.

**Key features:**
- 224×224 image processing
- Center-region detection with bounding box
- DSHOW backend for Windows compatibility
- 640×480 resolution for performance
- Live confidence percentage display
- Color-coded results (green=cat, red=dog)
- Keyboard quit control ('q' to exit)

**Usage**:
```powershell
python src/simple_detector.py
```

**Controls:**
- Center white box shows detection area
- Press 'q' to quit
- Results displayed in top-left corner

### Utility Scripts

#### `download_dataset.py`
Downloads the Dogs vs Cats dataset from Kaggle.

**Requirements**: Kaggle API credentials

**Usage**:
```powershell
python src/download_dataset.py
```

#### `balance_dataset.py`
Balances unequal cat/dog dataset samples.

**Features:**
- Calculates dataset statistics
- Removes excess samples from majority class
- Maintains train/validation split
- Preserves balanced proportions

**Usage**:
```powershell
python src/balance_dataset.py
```

#### `chart.py`
Generates performance visualization charts.

**Features:**
- Training/validation accuracy comparison
- Loss curves over epochs
- Comprehensive metrics dashboard
- PNG exports for reports

**Usage**:
```powershell
python src/chart.py
```

**Output files:**
- `1_metrics_comparison.png`
- `3_training_progress.png`
- `cat_dog_dashboard_2column.png`

## Model Information

### Architecture

The models use a Convolutional Neural Network (CNN) with:

- Input layer: 224×224×3 (RGB images)
- Multiple convolutional layers with ReLU activation
- Max pooling layers for feature extraction
- Dropout layers for regularization
- Dense layers for classification
- Output layer: Sigmoid activation (binary classification)

### Image Size

- **Training input**: 224×224 pixels
- **Color space**: RGB (converted from BGR)
- **Normalization**: Pixel values normalized to [0, 1]

### Model Files

| File | Accuracy | Format | Size | Notes |
|------|----------|--------|------|-------|
| `best_model_85.keras` | 85% | Keras | ~50MB | Best checkpoint |
| `cat_dog_classifier_85percent.keras` | 85% | Keras | ~50MB | Production model |
| `cat_dog_classifier.h5` | ~75-80% | H5 | ~50MB | Legacy format |

## Performance

### Metrics

- **Training Accuracy**: 92-95%
- **Validation Accuracy**: 85%
- **Test Accuracy**: 85%
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

### Training Configuration

- **Batch Size**: 32
- **Epochs**: 30-50
- **Learning Rate**: 0.001 (default)
- **Patience (Early Stopping)**: 5 epochs

## Troubleshooting

### Camera Issues

**Problem**: `Cannot open camera with DSHOW` error

**Solution**: 
```powershell
# Try using the alternative camera method in simple_camera.py
# The DSHOW backend is specifically for Windows
```

**Alternative**: Use `camera_detector.py` which has better error handling

### Dataset Issues

**Problem**: `Cannot find dataset folders`

**Solution**:
1. Ensure dataset is in `dogcat-env/dataset/`
2. Check folder structure matches requirements
3. Verify images are in correct subdirectories
4. Use `download_dataset.py` to auto-download

### Model Loading Issues

**Problem**: `Cannot load model` error

**Solution**:
```powershell
# Ensure model file exists in the correct location
# Try downloading the pre-trained model again
# Check file format (.keras vs .h5)
```

### Low Accuracy

**Problem**: Model predictions are inaccurate

**Solutions**:
1. Check if input images are properly formatted
2. Ensure images are 224×224 pixels
3. Verify normalization (values should be 0-1)
4. Retrain model with more epochs
5. Use data augmentation

## Advanced Usage

### Custom Training

Edit `train_model.py` to customize:

```python
# Change image size
img_height, img_width = 224, 224

# Adjust batch size
batch_size = 32

# Modify epochs
epochs = 50

# Change learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

### Transfer Learning

Use a pre-trained model instead of training from scratch:

```python
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(input_shape=(224, 224, 3), 
                         include_top=False, 
                         weights='imagenet')
```

### Model Export

Save model in different formats:

```python
# Keras format (recommended)
model.save('model.keras')

# H5 format (legacy)
model.save('model.h5')

# SavedModel format (TensorFlow serving)
model.save('model_savedmodel/')
```

## Dependencies

### Core Libraries

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.10 | Deep learning framework |
| keras | ≥2.10 | High-level API |
| numpy | ≥1.20 | Numerical computing |
| opencv-python | ≥4.5 | Computer vision |
| pillow | ≥8.0 | Image processing |
| matplotlib | ≥3.3 | Visualization |

### Optional Libraries

| Package | Purpose |
|---------|---------|
| scikit-learn | Dataset balancing |
| kagglehub | Dataset download from Kaggle |

## Contributing

To improve this project:

1. Test new architectures in `train_model.py`
2. Add new features to detection scripts
3. Document changes clearly
4. Update accuracy metrics
5. Share improvements

## Performance Tips

1. **GPU Acceleration**: Install `tensorflow-gpu` for faster training
2. **Batch Processing**: Process images in batches for better performance
3. **Image Preprocessing**: Use consistent preprocessing for predictions
4. **Model Optimization**: Use quantization for edge deployment
5. **Camera Resolution**: Lower resolution = faster predictions

## Future Enhancements

- [ ] Multi-class classification (cats, dogs, other animals)
- [ ] Fine-grained breed classification
- [ ] Model quantization for mobile deployment
- [ ] Web API integration
- [ ] Real-time statistics display
- [ ] Confidence threshold adjustment
- [ ] Batch image processing
- [ ] Model comparison dashboard

## License

This project is for educational purposes.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/shaunla/dogs-vs-cats)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review file comments in source code
3. Verify dataset structure
4. Test with pre-trained models first
5. Check console output for detailed error messages

---

**Last Updated**: March 2026
**Project Version**: 1.0
