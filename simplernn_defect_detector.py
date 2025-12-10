"""
SimpleRNN Defect Detection Model
Main implementation with training and evaluation functionality
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime


class SimpleRNNDefectDetector:
    def __init__(self, dataset_path, img_height=128, img_width=128, sequence_length=16):
        """
        Initialize SimpleRNN Defect Detection Model
        
        Args:
            dataset_path: Path to final_merged_dataset
            img_height: Height to resize images
            img_width: Width to resize images
            sequence_length: Number of feature vectors to treat as sequence
        """
        self.dataset_path = Path(dataset_path)
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
        self.class_names = []
        self.num_classes = 0
        
        # Create output directories
        self.output_dir = Path("rnn_output")
        self.model_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.model_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def prepare_data(self, validation_split=0.2, test_split=0.1, batch_size=32):
        """
        Prepare dataset with train/validation/test splits
        """
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Get all image paths and labels
        image_paths = []
        labels = []
        
        print("\n📁 Scanning dataset...")
        for class_dir in self.dataset_path.rglob("*"):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.bmp"))
                
                if images:
                    class_name = class_dir.name
                    if class_name not in self.class_names:
                        self.class_names.append(class_name)
                    
                    for img_path in images:
                        image_paths.append(str(img_path))
                        labels.append(class_name)
        
        self.num_classes = len(self.class_names)
        print(f"✓ Found {len(image_paths)} images across {self.num_classes} classes")
        print(f"✓ Classes: {self.class_names[:10]}{'...' if len(self.class_names) > 10 else ''}")
        
        # Convert labels to integers
        label_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        labels_encoded = [label_to_index[label] for label in labels]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels_encoded, test_size=test_split, random_state=42, stratify=labels_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split/(1-test_split), random_state=42, stratify=y_temp
        )
        
        print(f"\n✓ Train samples: {len(X_train)}")
        print(f"✓ Validation samples: {len(X_val)}")
        print(f"✓ Test samples: {len(X_test)}")
        
        # Create data generators
        self.train_generator = self._create_data_generator(X_train, y_train, batch_size, augment=True)
        self.val_generator = self._create_data_generator(X_val, y_val, batch_size, augment=False)
        self.test_generator = self._create_data_generator(X_test, y_test, batch_size, augment=False)
        
        self.steps_per_epoch = len(X_train) // batch_size
        self.validation_steps = len(X_val) // batch_size
        self.test_steps = len(X_test) // batch_size
        
        return len(X_train), len(X_val), len(X_test)
    
    def _create_data_generator(self, image_paths, labels, batch_size, augment=False):
        """Create a generator that yields batches of sequences"""
        def generator():
            indices = np.arange(len(image_paths))
            while True:
                np.random.shuffle(indices)
                
                for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    
                    batch_images = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        # Load and preprocess image
                        img = tf.keras.preprocessing.image.load_img(
                            image_paths[idx],
                            target_size=(self.img_height, self.img_width)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = img_array / 255.0  # Normalize
                        
                        # Data augmentation
                        if augment:
                            img_array = self._augment_image(img_array)
                        
                        batch_images.append(img_array)
                        batch_labels.append(labels[idx])
                    
                    # Convert to sequences for RNN
                    batch_images = np.array(batch_images)
                    batch_labels = np.array(batch_labels)
                    
                    # Reshape images to sequences: flatten spatial dimensions and create sequences
                    # Shape: (batch, height, width, channels) -> (batch, sequence_length, features)
                    sequences = self._image_to_sequence(batch_images)
                    
                    yield sequences, tf.keras.utils.to_categorical(batch_labels, self.num_classes)
        
        return generator()
    
    def _augment_image(self, img_array):
        """Apply random augmentations to image"""
        # Random flip
        if np.random.random() > 0.5:
            img_array = tf.image.flip_left_right(img_array)
        
        # Random brightness
        img_array = tf.image.random_brightness(img_array, max_delta=0.2)
        
        # Random contrast
        img_array = tf.image.random_contrast(img_array, lower=0.8, upper=1.2)
        
        return tf.clip_by_value(img_array, 0.0, 1.0)
    
    def _image_to_sequence(self, images):
        """
        Convert images to sequences for RNN processing
        Split image into horizontal strips to create a sequence
        """
        batch_size = images.shape[0]
        
        # Split image into sequence_length horizontal strips
        strip_height = self.img_height // self.sequence_length
        sequences = []
        
        for i in range(batch_size):
            img = images[i]
            strips = []
            
            for j in range(self.sequence_length):
                start_h = j * strip_height
                end_h = start_h + strip_height
                strip = img[start_h:end_h, :, :]
                # Flatten strip to create feature vector
                strip_flat = strip.reshape(-1)
                strips.append(strip_flat)
            
            sequences.append(strips)
        
        return np.array(sequences)
    
    def build_model(self, rnn_units=128, dropout_rate=0.3, learning_rate=0.001):
        """
        Build SimpleRNN model architecture
        """
        print("\n" + "="*60)
        print("BUILDING SIMPLERNN MODEL")
        print("="*60)
        
        # Calculate input feature size
        strip_height = self.img_height // self.sequence_length
        feature_size = strip_height * self.img_width * 3  # 3 for RGB channels
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, feature_size)),
            
            # First SimpleRNN layer with return sequences
            layers.SimpleRNN(rnn_units, return_sequences=True, activation='tanh'),
            layers.Dropout(dropout_rate),
            
            # Second SimpleRNN layer
            layers.SimpleRNN(rnn_units // 2, activation='tanh'),
            layers.Dropout(dropout_rate),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate / 2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n✓ Model architecture:")
        model.summary()
        
        # Save model architecture
        with open(self.model_dir / 'model_architecture.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        return model
    
    def train(self, epochs=50, early_stopping_patience=10):
        """
        Train the SimpleRNN model
        """
        print("\n" + "="*60)
        print("TRAINING SIMPLERNN MODEL")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                str(self.logs_dir / 'training_log.csv')
            )
        ]
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        print(f"⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save final model
        self.model.save(self.model_dir / 'final_model.h5')
        print(f"✓ Model saved to: {self.model_dir}")
        
        return self.history
    
    def evaluate(self):
        """
        Evaluate model on test set
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\n📊 Evaluating on test set...")
        test_results = self.model.evaluate(
            self.test_generator,
            steps=self.test_steps,
            verbose=1
        )
        
        print("\n✓ Test Results:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f}")
        print(f"  Top-5 Accuracy: {test_results[2]:.4f}")
        
        # Save results
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_top5_accuracy': float(test_results[2]),
            'num_classes': self.num_classes,
            'evaluation_time': datetime.now().isoformat()
        }
        
        with open(self.logs_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_training_history(self):
        """
        Plot training history
        """
        print("\n📈 Generating training plots...")
        
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot top-5 accuracy
        axes[1, 0].plot(self.history.history['top_5_accuracy'], label='Train Top-5 Acc')
        axes[1, 0].plot(self.history.history['val_top_5_accuracy'], label='Val Top-5 Acc')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                           ha='center', va='center', fontsize=14)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"✓ Training plots saved to: {self.plots_dir / 'training_history.png'}")
        plt.close()
    
    def save_training_summary(self):
        """
        Save comprehensive training summary
        """
        print("\n📝 Generating training summary...")
        
        summary_file = self.logs_dir / 'training_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SIMPLERNN DEFECT DETECTION - TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Dataset Path: {self.dataset_path}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Image Size: {self.img_height}x{self.img_width}\n")
            f.write(f"Sequence Length: {self.sequence_length}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*70 + "\n")
            f.write("Architecture: SimpleRNN with Dense layers\n")
            f.write(f"Total Parameters: {self.model.count_params():,}\n\n")
            
            if self.history:
                f.write("-"*70 + "\n")
                f.write("TRAINING RESULTS\n")
                f.write("-"*70 + "\n")
                f.write(f"Epochs Trained: {len(self.history.history['loss'])}\n")
                f.write(f"Best Train Accuracy: {max(self.history.history['accuracy']):.4f}\n")
                f.write(f"Best Val Accuracy: {max(self.history.history['val_accuracy']):.4f}\n")
                f.write(f"Final Train Loss: {self.history.history['loss'][-1]:.4f}\n")
                f.write(f"Final Val Loss: {self.history.history['val_loss'][-1]:.4f}\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"✓ Training summary saved to: {summary_file}")


def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("SIMPLERNN DEFECT DETECTION PIPELINE")
    print("="*60)
    
    # Configuration
    DATASET_PATH = "final_merged_dataset"
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    SEQUENCE_LENGTH = 16
    BATCH_SIZE = 32
    EPOCHS = 50
    RNN_UNITS = 128
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    
    # Initialize detector
    detector = SimpleRNNDefectDetector(
        dataset_path=DATASET_PATH,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Prepare data
    train_size, val_size, test_size = detector.prepare_data(
        validation_split=0.2,
        test_split=0.1,
        batch_size=BATCH_SIZE
    )
    
    # Build model
    detector.build_model(
        rnn_units=RNN_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    
    # Train model
    detector.train(epochs=EPOCHS, early_stopping_patience=10)
    
    # Evaluate model
    test_results = detector.evaluate()
    
    # Plot training history
    detector.plot_training_history()
    
    # Save summary
    detector.save_training_summary()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Output Directory: rnn_output/")
    print(f"📊 Models saved in: rnn_output/models/")
    print(f"📈 Plots saved in: rnn_output/plots/")
    print(f"📝 Logs saved in: rnn_output/logs/")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
