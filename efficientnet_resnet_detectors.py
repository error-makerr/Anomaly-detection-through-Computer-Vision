"""
EfficientNet and ResNet50 Defect Detection Models
Transfer learning implementations with base class
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class DefectDetectionModel:
    """Base class for defect detection models"""
    
    def __init__(self, dataset_path, model_name, img_height=224, img_width=224):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        self.class_names = []
        self.num_classes = 0
        
        # Create output directories
        self.output_dir = Path(f"{model_name}_output")
        self.model_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.model_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def prepare_data(self, validation_split=0.2, test_split=0.1, batch_size=32):
        """Prepare dataset with efficient TensorFlow pipeline"""
        print("\n" + "="*70)
        print(f"DATA PREPARATION - {self.model_name}")
        print("="*70)
        
        # Get all image paths and labels
        image_paths = []
        labels = []
        
        print("\n📁 Scanning dataset...")
        for class_dir in self.dataset_path.rglob("*"):
            if class_dir.is_dir():
                images = (list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + 
                         list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.bmp")))
                
                if images:
                    class_name = class_dir.name
                    if class_name not in self.class_names:
                        self.class_names.append(class_name)
                    
                    for img_path in images:
                        image_paths.append(str(img_path))
                        labels.append(class_name)
        
        self.class_names.sort()
        self.num_classes = len(self.class_names)
        print(f"✓ Found {len(image_paths)} images")
        print(f"✓ Number of classes: {self.num_classes}")
        print(f"✓ Classes: {self.class_names[:10]}{'...' if len(self.class_names) > 10 else ''}")
        
        # Create label mapping
        label_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        labels_encoded = [label_to_index[label] for label in labels]
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_encoded))
        dataset = dataset.shuffle(len(image_paths), seed=42)
        
        # Calculate splits
        train_size = int(len(image_paths) * (1 - validation_split - test_split))
        val_size = int(len(image_paths) * validation_split)
        test_size = len(image_paths) - train_size - val_size
        
        # Split datasets
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        test_ds = dataset.skip(train_size + val_size)
        
        # Process datasets
        self.train_dataset = train_ds.map(
            lambda x, y: self._load_and_preprocess(x, y, augment=True),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.val_dataset = val_ds.map(
            lambda x, y: self._load_and_preprocess(x, y, augment=False),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.test_dataset = test_ds.map(
            lambda x, y: self._load_and_preprocess(x, y, augment=False),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"\n✓ Train samples: {train_size}")
        print(f"✓ Validation samples: {val_size}")
        print(f"✓ Test samples: {test_size}")
        
        return train_size, val_size, test_size
    
    def _load_and_preprocess(self, image_path, label, augment=False):
        """Load and preprocess image"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        
        # Data augmentation
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        img = tf.clip_by_value(img, 0.0, 255.0)
        img = img / 255.0
        
        label_onehot = tf.one_hot(label, self.num_classes)
        return img, label_onehot
    
    def train(self, epochs=50, early_stopping_patience=10):
        """Train the model"""
        print("\n" + "="*70)
        print(f"TRAINING {self.model_name}")
        print("="*70)
        
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
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        print(f"⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save model weights
        self.model.save_weights(str(self.model_dir / 'model_weights.h5'))
        print(f"✓ Model weights saved to: {self.model_dir / 'model_weights.h5'}")
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print(f"EVALUATING {self.model_name}")
        print("="*70)
        
        print("\n📊 Evaluating on test set...")
        test_results = self.model.evaluate(self.test_dataset, verbose=1)
        
        print("\n✓ Test Results:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  Top-5 Accuracy: {test_results[2]:.4f}")
        
        # Get predictions for confusion matrix
        print("\n📊 Generating detailed metrics...")
        y_true = []
        y_pred = []
        
        for images, labels in self.test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # Save results
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'num_classes': self.num_classes,
            'evaluation_time': datetime.now().isoformat()
        }
        
        if len(test_results) > 2:
            results['test_top5_accuracy'] = float(test_results[2])
        
        with open(self.logs_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)
        
        # Generate classification report
        self._save_classification_report(y_true, y_pred)
        
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(max(10, self.num_classes), max(8, self.num_classes * 0.8)))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                    xticklabels=self.class_names if self.num_classes <= 20 else False,
                    yticklabels=self.class_names if self.num_classes <= 20 else False)
        
        plt.title(f'{self.model_name} - Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to: {self.plots_dir / 'confusion_matrix.png'}")
    
    def _save_classification_report(self, y_true, y_pred):
        """Save classification report"""
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                      zero_division=0)
        
        with open(self.logs_dir / 'classification_report.txt', 'w') as f:
            f.write(f"{self.model_name} - Classification Report\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"✓ Classification report saved to: {self.logs_dir / 'classification_report.txt'}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        print("\n📈 Generating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title(f'{self.model_name} - Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title(f'{self.model_name} - Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training plots saved to: {self.plots_dir / 'training_history.png'}")


class EfficientNetDefectDetector(DefectDetectionModel):
    """EfficientNet-B0 for defect detection"""
    
    def __init__(self, dataset_path, img_height=224, img_width=224):
        super().__init__(dataset_path, "EfficientNetB0", img_height, img_width)
    
    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        """Build EfficientNet model with transfer learning"""
        print("\n" + "="*70)
        print("BUILDING EFFICIENTNET MODEL")
        print("="*70)
        
        # Load pre-trained EfficientNet
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Scale from [0,1] to [0,255] for EfficientNet
        x = layers.Rescaling(scale=255.0)(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n✓ Model built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        print(f"✓ Base model frozen: {not base_model.trainable}")
        
        # Save architecture
        with open(self.model_dir / 'model_architecture.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        return model


class ResNet50DefectDetector(DefectDetectionModel):
    """ResNet-50 for defect detection"""
    
    def __init__(self, dataset_path, img_height=224, img_width=224):
        super().__init__(dataset_path, "ResNet50", img_height, img_width)
    
    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        """Build ResNet-50 model with transfer learning"""
        print("\n" + "="*70)
        print("BUILDING RESNET-50 MODEL")
        print("="*70)
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build model with preprocessing
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Use Keras preprocessing for ResNet (handles normalization)
        x = tf.keras.applications.resnet.preprocess_input(inputs * 255.0)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n✓ Model built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        print(f"✓ Base model frozen: {not base_model.trainable}")
        
        # Save architecture
        with open(self.model_dir / 'model_architecture.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        return model
