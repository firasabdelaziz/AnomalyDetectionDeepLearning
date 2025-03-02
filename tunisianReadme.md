# 🔍 Anomaly Detection bel Deep Learning

Salut les gars! Njim n9olek eli el projet hedha ye5dim 3la détection mta3 les choses li mahoumch normaux fil vidéos w yesta3mil el deep learning. Bech testa3mlou dataset UCSD eli fih vidéos surveillance.

## ✨ Chnowa fih el projet

- 📹 Preprocessing mta3 el vidéos bech na3mlou sequences lel training
- 🧠 Architecture hybride LSTM-CNN bech ye5ou features spatiales w temporelles
- 👁️ Visualisation mta3 el anomalies b contour detection
- 📊 Barcha metrics bech n9isou biha el performance (accuracy, precision, recall, w F1-score)
- 📈 ROC curve bech nchoufou 9adech el model behi

## 📂 Structure mta3 el projet

```
├── data/
│   └── UCSD_Anomaly_Dataset.v1p2/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Preprocessing mta3 el vidéos
│   ├── model.py             # Architecture mta3 el LSTM
│   └── evaluate.py          # Evaluation w visualisation
├── results/                 # El nateij w visualisations
├── best_model.h5            # El model ba3d el training
├── main.py                  # Script principal
└── README.md
```

## 🔧 Chnowa te7taj

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## 🚀 Installation

1. A3mil clone lel repository:
   ```
   git clone https://github.com/firasabdelaziz/AnomalyDetectionDeepLearning.git
   cd AnomalyDetectionDeepLearning
   ```

2. Instalili les packages:
   ```
   pip install tensorflow opencv-python numpy matplotlib scikit-learn
   ```

3. Téléchargili UCSD Anomaly Dataset:
   ```
   # Tnajem tnazzlou min houni:
   # http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
   
   # 7ottou fil dossier data:
   mkdir -p data
   # Extractlou fil data/UCSD_Anomaly_Dataset.v1p2/
   ```

## 💻 Kifech testa3mlou

### 🏋️ Training mta3 el Model

```python
from src.preprocessing import VideoPreprocessor
from src.model import AnomalyDetector

# Initialisi el preprocessor
preprocessor = VideoPreprocessor(frame_size=(32, 32), sequence_length=20)

# Prepari el data
X_train, X_test, y_train, y_test = preprocessor.prepare_data('data/UCSD_Anomaly_Dataset.v1p2')

# Initialisi w traini el model
detector = AnomalyDetector(20, 32, 32)
history = detector.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
```

### 🕵️ Testi el anomaly detection

```bash
# Exécuti el script principal
python main.py
```

El script bech:
1. Ya3mil load w preprocessing lel dataset
2. Ya3mil load lel model walla ya3mil training ken ma3andekch model
3. Ydetecti el anomalies fil test sequences
4. Ya3tik visualisation b contours 7awl el anomalies
5. Y9is el performance b barcha metrics
6. Ysajjel el visualisations fil dossier results

## 🏗️ Architecture mta3 el Model

El model yesta3mil architecture hybride CNN-LSTM:
- 🖼️ Layers CNN bech ye5dhou el features spatiales min kol frame
- ⏱️ Layers LSTM bech ye5dhou el patterns temporelles entre les frames
- 🧮 Layers Dense bech ya3mlou el classification finale

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

Na9isou el model b:
- ✓ **Accuracy**: 9adech el model sab s7i7 min el koll
- 🎯 **Precision**: 9adech min eli 9al 3lihom anomalies homma fعlan anomalies
- 🔍 **Recall**: 9adech min el anomalies el 7a9ania el model l9ahom
- 🔄 **F1-score**: Average harmonique bin precision w recall
- 📉 **ROC Curve**: Visualisation mta3 el true positive rate vs. false positive rate

## 🚀 Chnowa nziidou fil mosta9bel

- 🔄 Nizidou algoritmes o5rin mta3 anomaly detection (Autoencoders, GANs)
- ⚡ N3amlou support lel real-time video processing
- 🔥 N7assnou el visualisation b heatmaps bech nlocalisiw el anomalies
- 🌐 Nintegriw el projet m3a interface web bech nchoufou el nateij b tari9a ashel

## 📜 License

El projet ta7t MIT License - chouf el fichier LICENSE lel tafasel.

## 🙏 Chokr

- 🏫 UCSD 3la dataset mta3hom 
- 🧠 El team mta3 TensorFlow 3la el framework mta3hom el behi