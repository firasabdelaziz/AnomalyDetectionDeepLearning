import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import cv2
import os

class AnomalyEvaluator:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics."""
        y_pred_binary = (y_pred > self.threshold).astype(int)
        
        if len(np.unique(y_true)) == 1:
            print("Warning: y_true contains only one class. Precision, recall, and F1-score are undefined.")
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def plot_roc_curve(self, y_true, y_pred):
        """Plot the ROC curve."""
        if len(np.unique(y_true)) == 1:
            print("Warning: y_true contains only one class. ROC curve cannot be plotted.")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def run_evaluation(self, dataset_path, output_folder='results'):
        """Run evaluation on test videos and save results."""
        print(f"Evaluating test videos in {dataset_path}")
        test_path = os.path.join(dataset_path, 'UCSDped1', 'Test')
        os.makedirs(output_folder, exist_ok=True)
        
        for folder in sorted(glob.glob(os.path.join(test_path, "Test*"))):
            frames = []
            frame_files = sorted(glob.glob(os.path.join(folder, "*.tif")))
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    frame = cv2.resize(frame, (64, 64))  # Match CONFIG['frame_size']
                    frame = frame / 255.0
                    frames.append(frame)
            
            if len(frames) >= 20:  # Match CONFIG['sequence_length']
                sequences = []
                for i in range(len(frames) - 20 + 1):
                    sequence = frames[i:i + 20]
                    sequences.append(sequence)
                sequences = np.array(sequences)
                
                scores = self.model.predict(sequences)
                anomaly_mask = (scores > self.threshold).astype(np.uint8)
                
                # Save a sample frame with anomalies highlighted
                sample_frame = cv2.resize(frames[0] * 255, (256, 256)).astype(np.uint8)
                sample_mask = cv2.resize(anomaly_mask[0], (256, 256))
                sample_output = sample_frame.copy()
                contours, _ = cv2.findContours(sample_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(sample_output, contours, -1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(output_folder, f"{os.path.basename(folder)}_anomaly.jpg"), sample_output)