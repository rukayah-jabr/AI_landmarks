
# 🧠 AI Landmark Classification with Grad-CAM

This project is an AI-based landmark image classifier using transfer learning with InceptionV3 and visual explanation using Grad-CAM. Below is the setup and usage documentation so other contributors can run and test the system locally.

---

## 📦 1. Setup Instructions

### Clone and setup the environment

```bash
git clone <repository-url>
cd AI_landmarks
python -m venv venv310
venv310\Scripts\activate    # For Windows
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install manually:

```bash
pip install tensorflow keras opencv-python matplotlib
```

---

## 2. Project Structure Overview

```
AI_landmarks/
├── data/
│   └── landmarks/
│       ├── [Class1]/
│       ├── [Class2]/
│       └── ...
├── models/
│   └── landmark_model.h5          # Trained model file
├── scripts/
│   ├── gradcam.py                 # Grad-CAM visualization script
├── explore_data.py                # Sets the base_path used by other scripts
└── README.md
```

---

## 3. Model & Training

We use **InceptionV3** as the base model. The final layer is customized with 5 output classes (you can change this number in the `gradcam.py` script depending on your dataset).

The model is trained and saved to:

```
models/landmark_model.h5
```

---

## 4. Grad-CAM Explanation (scripts/gradcam.py)

This script:

1. Loads a saved model.
2. Selects a sample image from one class folder.
3. Predicts the class of the image.
4. Creates a Grad-CAM heatmap for visual explanation.
5. Overlays heatmap on the original image.
6. Displays and saves the output as `models/gradcam_result.png`.

It also prints:
- Predicted class index (e.g., 2)
- Predicted class label (e.g., “Stephansdom Vienna”)

To run:

```bash
python scripts/gradcam.py
```

Make sure `explore_data.py` contains a valid `base_path` pointing to the `data/landmarks` folder.

---

## 5. Script Descriptions

- **explore_data.py**  
  Contains the variable `base_path`, used to define the path to your landmarks dataset.

- **scripts/gradcam.py**  
  Performs Grad-CAM visualization on a single sample image using the trained model. Useful for debugging and explaining model predictions.

---

## 6. Common Issues

- FileNotFoundError: Check that your `base_path` is correct and folders like `data/landmarks/<ClassName>/` exist.
- Grad-CAM output shows black image: The input image might not be readable by OpenCV or path might be invalid.
- Wrong predictions: Model might require more training or better data preprocessing.

---

## 7. Sample Output

When successful, you will see:

```
Predicted class index: 3
Predicted class label: Belvedere Palace Vienna
```

And a Grad-CAM result image saved to:

```
models/gradcam_result.png
```

---

## 8. Contributors Setup Notes

Everyone should:

- Clone the repo
- Setup virtual environment and install dependencies
- Put class folders with images under `data/landmarks/`
- Run `gradcam.py` to test inference and visualization

For any questions, contact the project owner.

---
