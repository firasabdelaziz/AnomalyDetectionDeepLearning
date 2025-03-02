import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class AnomalyDetector:
    def __init__(self, sequence_length, frame_height, frame_width):
        from src.model import LSTMModel
        self.model = LSTMModel(sequence_length, frame_height, frame_width).build()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the model."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(patience=5),
                ModelCheckpoint('best_model.h5', save_best_only=True)
            ]
        )
        return history