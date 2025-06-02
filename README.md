# AI Landmark Classification with Grad-CAM

This project is an AI-based landmark image classifier using transfer learning with InceptionV3 and Grad-CAM for visual explanation. It allows you to classify landmark images and visualize where the model focused while making a prediction.

---

## 1. Setup Instructions

### Clone the repository and create a virtual environment

```bash
git clone https://github.com/rukayah-jabr/AI_landmarks.git
cd AI_landmarks
python -m venv venv310
venv310\Scripts\activate    # On Windows
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install the necessary libraries manually:

```bash
pip install tensorflow keras opencv-python matplotlib
```

---

## 2. Project Structure

```
AI_landmarks/
├── data/
│   └── landmarks/
│       ├── [Class1]/
│       ├── [Class2]/
│       └── ...
├── models/
│   └── landmark_model.h5      # (Not included in the repo, must be trained locally)
├── scripts/
│   ├── gradcam.py             # Grad-CAM visualization script
│   └── (train_model.py)       # Optional: create this to train the model
├── explore_data.py            # Sets the base_path used by scripts
└── README.md
```

---

## 3. Training the Model

The trained model file (`models/landmark_model.h5`) is **not included in this repository** because it exceeds GitHub's size limits.

You have two options:

### Option 1: Train it yourself

Ensure your training data is inside:

```
data/landmarks/
├── Class1/
├── Class2/
└── ...
```

Each class folder must contain images.

Then, use your own `train_model.py` script to train and save the model as:

```
models/landmark_model.h5
```

> Don't have a training script? Ask the project owner or refer to TensorFlow/Keras tutorials for `ImageDataGenerator` + `InceptionV3` fine-tuning.

---

## 4. Grad-CAM Visualization (`scripts/gradcam.py`)

This script:

1. Loads the trained model from `models/landmark_model.h5`
2. Picks a sample image from one of the class folders
3. Predicts the class of the image
4. Generates Grad-CAM heatmap
5. Overlays the heatmap on the image
6. Saves and displays the result

To run:

```bash
python scripts/gradcam.py
```

Make sure that `explore_data.py` contains a valid `base_path` pointing to the `data/landmarks` folder.

Output:

```
Predicted class index: 3
Predicted class label: Belvedere Palace Vienna
```

Image saved to:

```
models/gradcam_result.png
```

---

## 5. Script Descriptions

| Script                | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `explore_data.py`     | Defines `base_path` to your local dataset folder                       |
| `scripts/gradcam.py`  | Loads model, predicts a sample image, and shows Grad-CAM visualization |
| `scripts/train_model.py` *(optional)* | (You can create this) Train the InceptionV3 model on your dataset      |

---

## 6. Common Issues

- `FileNotFoundError`: Check `base_path` and make sure class folders exist
- Grad-CAM image is black: The image path may be invalid or unreadable
- Model not found: You must train and save `landmark_model.h5` locally
- Push failed: Virtual environment and model files are excluded from Git to avoid size issues

---

## 7. Contributors Guide

Each contributor should:

1. Clone the repo
2. Set up Python virtual environment
3. Install dependencies
4. Place class-based images in `data/landmarks/`
5. Train the model (if needed)
6. Run `scripts/gradcam.py` to test prediction and explanation

For any questions, contact the project owner.

---