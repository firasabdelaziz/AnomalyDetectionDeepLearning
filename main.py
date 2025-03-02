import os
import cv2
import numpy as np
import glob
from src.preprocessing import VideoPreprocessor
from src.model import LSTMModel
from src.evaluate import AnomalyEvaluator
import tensorflow as tf


def create_input_adapter(input_shape, output_shape):
    from tensorflow.keras.layers import Input, Lambda
    from tensorflow.keras.models import Model
    
    input_layer = Input(shape=input_shape)
    
    def resize_sequence(x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        reshaped = tf.reshape(x, [-1, input_shape[1], input_shape[2], input_shape[3]])
        resized = tf.image.resize(reshaped, [output_shape[1], output_shape[2]])
        return tf.reshape(resized, [batch_size, seq_len, output_shape[1], output_shape[2], output_shape[3]])
    
    output_layer = Lambda(resize_sequence)(input_layer)
    return Model(inputs=input_layer, outputs=output_layer)


def visualize_anomalies(image, anomaly_mask):
    """Draw contours around anomalies in the image."""
    if anomaly_mask.ndim != 2:
        raise ValueError(f"anomaly_mask must be a 2D array, but got shape {anomaly_mask.shape}")
    
    anomaly_mask = (anomaly_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)
    return image_with_contours


def main():
    # 1. Set up configurations
    CONFIG = {
        'frame_size': (32, 32),
        'sequence_length': 20,
        'epochs': 20,
        'batch_size': 32,
        'dataset_path': 'data/UCSD_Anomaly_Dataset.v1p2',
        'test_folder': 'data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test034',
        'best_model_path': 'best_model.h5',
        'results_folder': 'results'
    }
    
    os.makedirs(CONFIG['results_folder'], exist_ok=True)
    
    # 2. Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = VideoPreprocessor(frame_size=CONFIG['frame_size'], sequence_length=CONFIG['sequence_length'])
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(CONFIG['dataset_path'])
    
    print("Total sequences processed:", len(X_train) + len(X_test))
    print("y_train unique values:", np.unique(y_train))
    print("y_test unique values:", np.unique(y_test))
    
    # 3. Load the trained model
    if os.path.exists(CONFIG['best_model_path']):
        print("Loading pre-trained model...")
        original_model = tf.keras.models.load_model(CONFIG['best_model_path'])
        
        input_adapter = create_input_adapter(
            input_shape=(CONFIG['sequence_length'], CONFIG['frame_size'][0], CONFIG['frame_size'][1], 3),
            output_shape=(CONFIG['sequence_length'], 64, 64, 3)
        )
    else:
        raise FileNotFoundError(f"Model not found at {CONFIG['best_model_path']}. Train the model first.")
    
    # 4. Load and preprocess test images
    print(f"Loading test images from: {CONFIG['test_folder']}")
    test_images = []
    for frame_file in sorted(glob.glob(os.path.join(CONFIG['test_folder'], "*.tif"))):
        frame = cv2.imread(frame_file)
        if frame is not None:
            frame = cv2.resize(frame, CONFIG['frame_size'])
            frame = frame / 255.0
            frame = frame.astype(np.float32)
            test_images.append(frame)
    
    if not test_images:
        raise ValueError(f"No images found in {CONFIG['test_folder']}. Please check the folder path.")
    
    test_images = np.array(test_images)
    print("Test images shape:", test_images.shape)
    
    print("Splitting test images into sequences...")
    test_sequences = []
    for i in range(len(test_images) - CONFIG['sequence_length'] + 1):
        sequence = test_images[i:i + CONFIG['sequence_length']]
        test_sequences.append(sequence)
    test_sequences = np.array(test_sequences)
    print("Test sequences shape:", test_sequences.shape)
    
    # 5. Predict anomalies
    print("Predicting anomalies...")
    anomaly_scores = []
    for sequence in test_sequences:
        sequence = sequence[np.newaxis, ...]
        print("Sequence shape before adaptation:", sequence.shape)
        adapted_sequence = input_adapter.predict(sequence, verbose=0)
        print("Adapted sequence shape:", adapted_sequence.shape)
        scores = original_model.predict(adapted_sequence, verbose=0)
        anomaly_scores.append(scores.squeeze())
    anomaly_scores = np.array(anomaly_scores)
    print("Sample anomaly scores:", anomaly_scores[:5])
    
    # Thresholding
    threshold = 0.5
    anomaly_mask = (anomaly_scores > threshold).astype(np.uint8)
    print("Anomaly mask shape:", anomaly_mask.shape)
    
    # 6. Visualize anomalies
    #print("Visualizing anomalies...")
    #for i, sequence in enumerate(test_sequences):
    #    # Create a 2D mask for the entire sequence based on the anomaly score
    #    sequence_anomaly_value = anomaly_mask[i]  # 0 or 1
    #    # Create a 2D mask with the same value across all pixels
    #    frame_mask = np.full((32, 32), sequence_anomaly_value, dtype=np.float32)
    #    
    #    for j, frame in enumerate(sequence):
    #        original_frame = cv2.resize(frame, (frame.shape[1] * 4, frame.shape[0] * 4))
    #        # Resize the mask to match the upscaled frame
    #        frame_anomaly_mask = cv2.resize(
    #            frame_mask,
    #            (original_frame.shape[1], original_frame.shape[0]),
    #            interpolation=cv2.INTER_NEAREST
    #        )
    #        frame_with_contours = visualize_anomalies(original_frame, frame_anomaly_mask)
    #        cv2.imshow("Anomaly Detection", frame_with_contours)
    #        cv2.waitKey(100)
    #cv2.destroyAllWindows()
    
    # 7. Evaluate model
    print("Evaluating model...")
    evaluator = AnomalyEvaluator(original_model)
    X_test_adapted = input_adapter.predict(X_test, verbose=0)
    y_pred = original_model.predict(X_test_adapted, verbose=0)
    print("Sample y_pred:", y_pred[:5])
    metrics = evaluator.compute_metrics(y_test, y_pred)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    evaluator.plot_roc_curve(y_test, y_pred)
    
    # 8. Process test videos
    print("\nProcessing test videos...")
    evaluator.run_evaluation(dataset_path=CONFIG['dataset_path'], output_folder=CONFIG['results_folder'])
    
    print(f"\nComplete! Check the results in the {CONFIG['results_folder']} directory.")


if __name__ == "__main__":
    main()