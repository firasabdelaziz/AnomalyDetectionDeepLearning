import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

class VideoPreprocessor:
    def __init__(self, frame_size=(32, 32), sequence_length=20):
        self.frame_size = frame_size
        self.sequence_length = sequence_length
    
    def load_dataset_generator(self, dataset_path):
        """Generator to yield sequences and labels one at a time."""
        print(f"Loading dataset from: {dataset_path}")
        
        train_path = os.path.join(dataset_path, 'UCSDped1', 'Train')
        test_path = os.path.join(dataset_path, 'UCSDped1', 'Test')
        
        # Process training data (normal sequences)
        print("Processing training data...")
        if os.path.exists(train_path):
            train_folders = sorted(glob.glob(os.path.join(train_path, "Train*")))
            for folder in train_folders:
                frames = []
                frame_files = sorted(glob.glob(os.path.join(folder, "*.tif")))
                for frame_file in frame_files:
                    frame = cv2.imread(frame_file)
                    if frame is not None:
                        frame = cv2.resize(frame, self.frame_size)
                        frame = frame / 255.0  # Normalize
                        frame = frame.astype(np.float32)  # Use float32
                        frames.append(frame)
                
                if len(frames) >= self.sequence_length:
                    for i in range(len(frames) - self.sequence_length + 1):
                        sequence = frames[i:i + self.sequence_length]
                        yield np.array(sequence), 0  # Normal sequence
        
        # Process test data (mix of normal and anomalous sequences)
        print("Processing test data...")
        if os.path.exists(test_path):
            test_folders = sorted(glob.glob(os.path.join(test_path, "Test*")))
            for folder in test_folders:
                frames = []
                frame_files = sorted(glob.glob(os.path.join(folder, "*.tif")))
                for frame_file in frame_files:
                    frame = cv2.imread(frame_file)
                    if frame is not None:
                        frame = cv2.resize(frame, self.frame_size)
                        frame = frame / 255.0
                        frame = frame.astype(np.float32)
                        frames.append(frame)
                
                if len(frames) >= self.sequence_length:
                    for i in range(len(frames) - self.sequence_length + 1):
                        sequence = frames[i:i + self.sequence_length]
                        label = 1 if 'Test034' in folder else 0  # Example heuristic
                        yield np.array(sequence), label
    
    def prepare_data(self, dataset_path):
        """Prepare data for training and testing using a list to conserve memory."""
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        sequences = []
        labels = []
        for sequence, label in self.load_dataset_generator(dataset_path):
            sequences.append(sequence)
            labels.append(label)
        
        print(f"Total sequences processed: {len(sequences)}")
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)