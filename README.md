# Face Mask Detection

A deep learning project that uses Convolutional Neural Networks (CNN) to automatically detect whether a person in an image or video is wearing a face mask or not.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Face Mask Detection project aims to develop an intelligent system that can identify whether individuals are wearing face masks in images or video streams. This is particularly useful for:

- Public health monitoring
- Safety compliance in workplaces
- Automated surveillance systems
- Mobile and edge device deployment

The project leverages Convolutional Neural Networks (CNN) for robust and accurate detection capabilities.

## ✨ Features

- **Image Detection**: Analyze static images to detect face masks
- **Video Detection**: Real-time detection in video streams
- **High Accuracy**: CNN-based model trained on diverse datasets
- **Easy Integration**: Simple API for integration into other applications
- **Pre-trained Models**: Ready-to-use trained models included

## 📊 Dataset

The model is trained on a comprehensive dataset containing:
- Images of people **with** face masks
- Images of people **without** face masks
- Various lighting conditions and angles
- Different mask types and styles

## 🧠 Model Architecture

The project utilizes a Convolutional Neural Network with the following characteristics:

- **Backbone**: Deep CNN architecture for feature extraction
- **Layers**: Multiple convolutional and pooling layers
- **Output**: Binary classification (mask/no mask)
- **Optimization**: Adam optimizer with categorical cross-entropy loss
- **Augmentation**: Data augmentation techniques to improve robustness

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook
- Required libraries: TensorFlow, Keras, OpenCV, NumPy, Pandas, Scikit-learn

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Kif30/face-mask-detection.git
cd face-mask-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 💻 Usage

### Running the Notebooks

1. **Training**: Open and run the training notebook to train the model on your dataset
2. **Evaluation**: Use the evaluation notebook to test model performance
3. **Inference**: Run the inference notebook for predictions on new images/videos

### Example Code

```python
# Load the trained model
from tensorflow import keras
model = keras.models.load_model('model_path.h5')

# Make predictions
predictions = model.predict(image)
```

## 📈 Results

The trained model achieves:
- High accuracy on test datasets
- Fast inference time suitable for real-time applications
- Robust performance across various conditions

(Detailed metrics can be found in the notebooks)

## 📁 Project Structure

```
face-mask-detection/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   └── inference.ipynb
├── models/
│   └── trained_model.h5
├── data/
│   ├── with_mask/
│   └── without_mask/
└── src/
    ├── model.py
    ├── utils.py
    └── preprocessing.py
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is available under the MIT License - see the LICENSE file for details.

---

**Language Composition:**
- Jupyter Notebook: 98.6%
- Python: 1.4%

**Author**: Kif30  
**Repository**: [Kif30/face-mask-detection](https://github.com/Kif30/face-mask-detection)
