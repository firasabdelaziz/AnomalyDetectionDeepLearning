# model.py
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten

class LSTMModel:
    def __init__(self, sequence_length, frame_height, frame_width):
        self.sequence_length = sequence_length
        self.frame_height = frame_height
        self.frame_width = frame_width
        
    def build(self):
        """Build the LSTM model architecture"""
        model = Sequential([
            # CNN layers for spatial features
            TimeDistributed(Conv2D(32, (3, 3), activation='relu', 
                          input_shape=(self.sequence_length, self.frame_height, self.frame_width, 3))),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Flatten()),
            
            # LSTM layers for temporal features
            LSTM(256, return_sequences=True),
            LSTM(128),
            
            # Dense layers for classification
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        return model
    
    def load_trained_model(self, model_path):
        """Load a trained model from disk"""
        return load_model(model_path)