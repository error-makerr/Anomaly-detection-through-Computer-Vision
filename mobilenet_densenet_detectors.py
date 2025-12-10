"""
MobileNetV2 and DenseNet Defect Detection Models
Lightweight and dense architectures for efficient training
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, DenseNet121, DenseNet169, DenseNet201
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
        
        self.output_dir = Path(f"{model_name}_output")
        self.model_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.model_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def prepare_data(self, validation_split=0.2, test_split=0.1, batch_size=32):
        """Prepare dataset"""
        print("\n" + "="*70)
        print(f"DATA PREPARATION - {self.model_name}")
        print("="*70)
        
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
        print(f"✓ Found {len(image_paths)} images across {self.num_classes} classes")
        
        label_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        labels_encoded = [label_to_index[label] for label in labels]
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_encoded))
        dataset = dataset.shuffle(len(image_paths), seed=42)
        
        train_size = int(len(image_paths) * (1 - validation_split - test_split))
        val_size = int(len(image_paths) * validation_split)
        test_size = len(image_paths) - train_size - val_size
        
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        test_ds = dataset.skip(train_size + val_size)
        
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
        
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        
        self.model.save_weights(str(self.model_dir / 'model_weights.h5'))
        print(f"✓ Model weights saved")
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print(f"EVALUATING {self.model_name}")
        print("="*70)
        
        test_results = self.model.evaluate(self.test_dataset, verbose=1)
        
        print("\n✓ Test Results:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  Top-5 Accuracy: {test_results[2]:.4f}")
        
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
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        print("\n📈 Generating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title(f'{self.model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title(f'{self.model_name} - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training plots saved")


class MobileNetV2DefectDetector(DefectDetectionModel):
    """MobileNetV2 for defect detection - Lightweight"""
    
    def __init__(self, dataset_path, img_height=224, img_width=224):
        super().__init__(dataset_path, "MobileNetV2", img_height, img_width)
    
    def build_model(self, learning_rate=0.001, dropout_rate=0.3, alpha=1.0):
        print("\n" + "="*70)
        print("BUILDING MOBILENETV2 MODEL")
        print("="*70)
        
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3),
            alpha=alpha
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n✓ Model built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        
        return model


class DenseNetDefectDetector(DefectDetectionModel):
    """DenseNet for defect detection"""
    
    def __init__(self, dataset_path, model_variant='121', img_height=224, img_width=224):
        self.model_variant = model_variant
        super().__init__(dataset_path, f"DenseNet{model_variant}", img_height, img_width)
    
    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        print("\n" + "="*70)
        print(f"BUILDING DENSENET{self.model_variant} MODEL")
        print("="*70)
        
        if self.model_variant == '121':
            base_model = DenseNet121(include_top=False, weights='imagenet', 
                                    input_shape=(self.img_height, self.img_width, 3))
        elif self.model_variant == '169':
            base_model = DenseNet169(include_top=False, weights='imagenet', 
                                    input_shape=(self.img_height, self.img_width, 3))
        elif self.model_variant == '201':
            base_model = DenseNet201(include_top=False, weights='imagenet', 
                                    input_shape=(self.img_height, self.img_width, 3))
        else:
            raise ValueError("Variant must be 121, 169 or 201")
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = tf.keras.applications.densenet.preprocess_input(inputs * 255.0)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate/2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate/4)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n✓ Model built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        
        return model


def train_mobilenetv2(dataset_path="final_merged_dataset", epochs=30, batch_size=32, alpha=1.0):
    """Train MobileNetV2 model"""
    print("\n\n" + "📱"*35)
    print("TRAINING MOBILENETV2")
    print("📱"*35)
    
    model = MobileNetV2DefectDetector(dataset_path)
    model.prepare_data(batch_size=batch_size)
    model.build_model(learning_rate=0.001, alpha=alpha)
    model.train(epochs=epochs)
    results = model.evaluate()
    model.plot_training_history()
    
    return results


def train_densenet(dataset_path="final_merged_dataset", variant='121', epochs=30, batch_size=32):
    """Train DenseNet model"""
    print("\n\n" + "🌳"*35)
    print(f"TRAINING DENSENET{variant}")
    print("🌳"*35)
    
    model = DenseNetDefectDetector(dataset_path, model_variant=variant)
    model.prepare_data(batch_size=batch_size)
    model.build_model(learning_rate=0.001)
    model.train(epochs=epochs)
    results = model.evaluate()
    model.plot_training_history()
    
    return results
