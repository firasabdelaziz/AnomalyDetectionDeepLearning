# 🔍 Anomaly Detection in Video with Deep Learning

This project implements a deep learning approach to detect anomalies in video sequences using an LSTM-based model. It's designed to identify unusual patterns or behaviors in surveillance footage from the UCSD Anomaly Dataset.

## ✨ Features

- 📹 Video preprocessing pipeline for creating sequence-based training data
- 🧠 LSTM-CNN hybrid architecture for spatiotemporal feature learning
- 👁️ Anomaly visualization with contour detection
- 📊 Comprehensive evaluation metrics including accuracy, precision, recall, and F1-score
- 📈 ROC curve visualization for model performance analysis

## 📂 Project Structure

```
├── data/
│   └── UCSD_Anomaly_Dataset.v1p2/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Video sequence preprocessing
│   ├── model.py             # LSTM model architecture
│   └── evaluate.py          # Evaluation metrics and visualization
├── results/                 # Generated results and visualizations
├── best_model.h5            # Trained model weights
├── main.py                  # Main execution script
└── README.md
```

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## 🚀 Installation

1. Clone this repository:
   ```
   git clone https://github.com/firasabdelaziz/AnomalyDetectionDeepLearning.git
   cd AnomalyDetectionDeepLearning
   ```

2. Install the required packages:
   ```
   pip install tensorflow opencv-python numpy matplotlib scikit-learn
   ```

3. Download the UCSD Anomaly Dataset:
   ```
   # The dataset can be downloaded from:
   # http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
   
   # Place it in the data directory:
   mkdir -p data
   # Extract the dataset to data/UCSD_Anomaly_Dataset.v1p2/
   ```

## 💻 Usage

### 🏋️ Training the Model

```python
from src.preprocessing import VideoPreprocessor
from src.model import AnomalyDetector

# Initialize the preprocessor
preprocessor = VideoPreprocessor(frame_size=(32, 32), sequence_length=20)

# Prepare the data
X_train, X_test, y_train, y_test = preprocessor.prepare_data('data/UCSD_Anomaly_Dataset.v1p2')

# Initialize and train the model
detector = AnomalyDetector(20, 32, 32)
history = detector.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
```

### 🕵️ Running Anomaly Detection

```bash
# Run the main script
python main.py
```

This will:
1. Load and preprocess the video dataset
2. Load the trained model or train a new one if no model exists
3. Detect anomalies in test sequences
4. Visualize the results with contours around detected anomalies
5. Evaluate model performance with various metrics
6. Save visualizations to the results folder

## 🏗️ Model Architecture

The model uses a hybrid CNN-LSTM architecture:
- 🖼️ CNN layers extract spatial features from each frame
- ⏱️ LSTM layers capture temporal patterns across frame sequences
- 🧮 Dense layers perform final classification

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
time_distributed (TimeDistri (None, 20, 30, 30, 32)    896       
_________________________________________________________________
time_distributed_1 (TimeDist (None, 20, 15, 15, 32)    0         
_________________________________________________________________
time_distributed_2 (TimeDist (None, 20, 13, 13, 64)    18496     
_________________________________________________________________
time_distributed_3 (TimeDist (None, 20, 6, 6, 64)      0         
_________________________________________________________________
time_distributed_4 (TimeDist (None, 20, 2304)          0         
_________________________________________________________________
lstm (LSTM)                  (None, 20, 256)           2621440   
_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               197120    
_________________________________________________________________
dense (Dense)                (None, 64)                8256      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 2,846,273
Trainable params: 2,846,273
Non-trainable params: 0
```

## 📏 Evaluation

The model is evaluated using:
- ✓ **Accuracy**: Overall correct classification rate
- 🎯 **Precision**: Ratio of true anomalies to all detected anomalies
- 🔍 **Recall**: Ratio of detected anomalies to all actual anomalies
- 🔄 **F1-score**: Harmonic mean of precision and recall
- 📉 **ROC Curve**: Visual representation of the true positive rate vs. false positive rate

## 🚀 Future Improvements

- 🔄 Implement additional anomaly detection algorithms (Autoencoders, GANs)
- ⚡ Add support for real-time video processing
- 🔥 Improve visualization with heatmaps for anomaly localization
- 🌐 Integrate with a web interface for easier exploration of results

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- 🏫 UCSD for providing the Anomaly Dataset
- 🧠 The TensorFlow team for their excellent deep learning framework