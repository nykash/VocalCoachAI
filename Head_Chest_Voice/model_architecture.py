import numpy as np
import librosa
import librosa.display
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import cv2
import pickle
import os
import zipfile
import requests
from tqdm import tqdm
import glob



class VocalRegisterClassifier:
    """
    SVM-based vocal register classifier based on mel-spectrogram analysis
    Following the methodology from arXiv:2505.11378
    """
    
    def __init__(self, img_height=154, img_width=128):
        self.img_height = img_height
        self.img_width = img_width
        self.scaler = StandardScaler()
        self.svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        
    def audio_to_melspec(self, audio, sr=22050):
        """
        Convert audio to mel-spectrogram following paper specifications
        """
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=128,
            fmax=8000
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def preprocess_spectrogram(self, mel_spec_db, augment=False):
        """
        Preprocess mel-spectrogram to fixed size image
        Based on paper: resize to 154x128, convert to grayscale
        """
        # Normalize to 0-255 range for image processing
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                        (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # Resize to target dimensions
        img_resized = cv2.resize(mel_spec_norm, (self.img_width, self.img_height))
        
        images = [img_resized]
        
        # Data augmentation as per paper
        if augment:
            # Horizontal flip
            img_flipped = cv2.flip(img_resized, 1)
            images.append(img_flipped)
            
            # Brightness adjustments (0.8x and 1.2x)
            img_dark = np.clip(img_resized * 0.8, 0, 255).astype(np.uint8)
            img_bright = np.clip(img_resized * 1.2, 0, 255).astype(np.uint8)
            images.append(img_dark)
            images.append(img_bright)
        
        return images
    
    def extract_features(self, img):
        """
        Flatten image to feature vector (19,712 dimensions = 154 x 128)
        """
        return img.flatten()
    
    def load_audio_file(self, filepath, sr=22050):
        """
        Load audio file and handle errors
        """
        try:
            audio, _ = librosa.load(filepath, sr=sr)
            return audio
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def prepare_dataset_from_files(self, data_dir, augment_train=True, train_split=0.7, val_split=0.15):
        """
        Prepare dataset from downloaded audio files
        """
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        print(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        # Shuffle files
        np.random.seed(42)
        np.random.shuffle(audio_files)
        
        # Split into train/val/test
        n_train = int(len(audio_files) * train_split)
        n_val = int(len(audio_files) * val_split)
        
        train_files = audio_files[:n_train]
        val_files = audio_files[n_train:n_train + n_val]
        test_files = audio_files[n_train + n_val:]
        
        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Process each split
        for split_name, file_list, X_list, y_list, do_augment in [
            ('train', train_files, X_train, y_train, augment_train),
            ('validation', val_files, X_val, y_val, False),
            ('test', test_files, X_test, y_test, False)
        ]:
            print(f"\nProcessing {split_name} split...")
            
            for idx, filepath in enumerate(tqdm(file_list, desc=split_name)):
                # Determine label from filename
                filename = os.path.basename(filepath).lower()
                
                # Label: 0 = chest, 1 = falsetto/head
                if 'falsetto' in filename or 'head' in filename:
                    label = 1
                elif 'chest' in filename:
                    label = 0
                else:
                    # Try to infer from parent directory
                    parent = os.path.basename(os.path.dirname(filepath)).lower()
                    if 'falsetto' in parent or 'head' in parent:
                        label = 1
                    elif 'chest' in parent:
                        label = 0
                    else:
                        print(f"  Warning: Cannot determine label for {filepath}, skipping")
                        continue
                
                # Load audio
                audio = self.load_audio_file(filepath)
                if audio is None:
                    continue
                
                # Convert to mel-spectrogram
                mel_spec = self.audio_to_melspec(audio)
                
                # Preprocess and optionally augment
                images = self.preprocess_spectrogram(mel_spec, augment=do_augment)
                
                # Extract features from each image
                for img in images:
                    features = self.extract_features(img)
                    X_list.append(features)
                    y_list.append(label)
        
        # Combine validation and test
        X_test_combined = X_val + X_test
        y_test_combined = y_val + y_test
        
        print(f"\nDataset preparation complete!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test_combined)}")
        print(f"Chest voice in training: {sum(1 for y in y_train if y == 0)}")
        print(f"Head voice in training: {sum(1 for y in y_train if y == 1)}")
        
        return (np.array(X_train), np.array(y_train), 
                np.array(X_test_combined), np.array(y_test_combined))
    
    def train(self, X_train, y_train):
        """
        Train the SVM classifier
        """
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training SVM...")
        self.svm.fit(X_train_scaled, y_train)
        
        print("Training complete!")
        
    def predict(self, audio, sr=22050):
        """
        Predict vocal register for a given audio segment
        Returns: (prediction, probability)
        """
        # Convert to mel-spectrogram
        mel_spec = self.audio_to_melspec(audio, sr)
        
        # Preprocess
        images = self.preprocess_spectrogram(mel_spec, augment=False)
        
        # Extract features
        features = self.extract_features(images[0]).reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.svm.predict(features_scaled)[0]
        probabilities = self.svm.predict_proba(features_scaled)[0]
        
        label = "Head Voice/Falsetto" if prediction == 1 else "Chest Voice"
        confidence = probabilities[prediction]
        
        return label, confidence
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.svm.predict(X_test_scaled)
        
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Chest Voice', 'Head Voice/Falsetto']))
        
        return accuracy_score(y_test, y_pred)
    
    def save_model(self, filepath='vocal_register_classifier.pkl'):
        """
        Save trained model and scaler
        """
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'img_height': self.img_height,
            'img_width': self.img_width
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='vocal_register_classifier.pkl'):
        """
        Load trained model and scaler
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.img_height = model_data['img_height']
        self.img_width = model_data['img_width']
        print(f"Model loaded from {filepath}")


# ============== DATASET DOWNLOAD ==============

def download_file(url, output_path):
    """
    Download file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_and_extract_dataset(data_dir='./chest_falsetto_data'):
    """
    Download and extract the chest/falsetto dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset URLs (from ModelScope mirror since HuggingFace has issues)
    audio_url = "https://www.modelscope.cn/datasets/ccmusic-database/chest_falsetto/resolve/master/data/audio.zip"
    
    audio_zip_path = os.path.join(data_dir, 'audio.zip')
    
    # Download if not already downloaded
    if not os.path.exists(audio_zip_path):
        print("Downloading audio dataset...")
        download_file(audio_url, audio_zip_path)
    else:
        print("Audio dataset already downloaded")
    
    # Extract
    audio_extract_path = os.path.join(data_dir, 'audio')
    if not os.path.exists(audio_extract_path):
        print("Extracting audio files...")
        with zipfile.ZipFile(audio_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete")
    else:
        print("Audio files already extracted")
    
    return data_dir


# ============== TRAINING SCRIPT ==============

def train_classifier(data_dir='./chest_falsetto_data'):
    """
    Complete training pipeline
    """
    print("=" * 60)
    print("DOWNLOADING AND PREPARING DATASET")
    print("=" * 60)
    
    # Download and extract dataset
    data_dir = download_and_extract_dataset(data_dir)
    
    # Initialize classifier
    classifier = VocalRegisterClassifier()
    
    # Prepare dataset from files
    print("\n" + "=" * 60)
    print("PREPARING DATASET WITH AUGMENTATION")
    print("=" * 60)
    X_train, y_train, X_test, y_test = classifier.prepare_dataset_from_files(data_dir)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    classifier.train(X_train, y_train)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    accuracy = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save_model('vocal_register_svm.pkl')
    
    return classifier


# ============== REAL-TIME INFERENCE ==============

def real_time_classification(audio_file, model_path='vocal_register_svm.pkl', 
                            segment_length=2.0, hop_length=0.5):
    """
    Classify vocal register in real-time for an audio file
    """
    # Load model
    classifier = VocalRegisterClassifier()
    classifier.load_model(model_path)
    
    # Load audio
    print(f"\nLoading audio file: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=22050)
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")
    
    # Process in segments
    segment_samples = int(segment_length * sr)
    hop_samples = int(hop_length * sr)
    
    results = []
    
    print(f"\nProcessing segments (length={segment_length}s, hop={hop_length}s)...")
    for i in range(0, len(audio) - segment_samples, hop_samples):
        segment = audio[i:i + segment_samples]
        label, confidence = classifier.predict(segment, sr)
        
        time_start = i / sr
        time_end = (i + segment_samples) / sr
        results.append({
            'time_start': time_start,
            'time_end': time_end,
            'label': label,
            'confidence': confidence
        })
        
        print(f"  {time_start:6.2f}s - {time_end:6.2f}s: {label:20s} (confidence: {confidence:.1%})")
    
    return results


# ============== USAGE EXAMPLE ==============

if __name__ == "__main__":
    # Training
    print("=" * 60)
    print("TRAINING VOCAL REGISTER CLASSIFIER")
    print("=" * 60)
    classifier = train_classifier()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTo use the model for classification, run:")
    print("  results = real_time_classification('path/to/your_audio.mp3')")
    
    # Uncomment below to test on a file
    # print("\n" + "=" * 60)
    # print("REAL-TIME CLASSIFICATION EXAMPLE")
    # print("=" * 60)
    # results = real_time_classification('path/to/vocal.mp3')