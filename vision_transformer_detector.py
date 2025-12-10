"""Vision Transformer (ViT) for Defect Detection"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class VisionTransformerDefectDetector:
    """Vision Transformer (ViT) for defect detection"""
    
    def __init__(self, dataset_path, img_height=224, img_width=224, patch_size=16):
        self.dataset_path = Path(dataset_path)
        self.model_name = "VisionTransformer"
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        self.model = None
        self.history = None
        self.class_names = []
        self.num_classes = 0
        
        # Create output directories
        self.output_dir = Path("VisionTransformer_output")
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
        img = img / 255.0  # Normalize to [0, 1]
        
        label_onehot = tf.one_hot(label, self.num_classes)
        return img, label_onehot
    
    def build_model(self, transformer_layers=8, num_heads=8, mlp_dim=2048, 
                    dropout_rate=0.1, learning_rate=0.001):
        """Build Vision Transformer model"""
        print("\n" + "="*70)
        print("BUILDING VISION TRANSFORMER MODEL")
        print("="*70)
        
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Create patches
        patches = self._create_patches(inputs)
        
        # Patch embedding
        projection_dim = 768
        patch_embeddings = layers.Dense(projection_dim)(patches)
        
        # Position embedding
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = layers.Embedding(
            input_dim=self.num_patches, output_dim=projection_dim
        )(positions)
        
        # Add position embeddings
        encoded = patch_embeddings + position_embeddings
        
        # Transformer blocks
        for _ in range(transformer_layers):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=dropout_rate
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, encoded])
            
            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = layers.Dense(mlp_dim, activation='gelu')(x3)
            x3 = layers.Dropout(dropout_rate)(x3)
            x3 = layers.Dense(projection_dim)(x3)
            x3 = layers.Dropout(dropout_rate)(x3)
            
            # Skip connection 2
            encoded = layers.Add()([x3, x2])
        
        # Final layer normalization
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
        
        # Global average pooling
        representation = layers.GlobalAveragePooling1D()(representation)
        
        # Classification head
        representation = layers.Dropout(dropout_rate)(representation)
        features = layers.Dense(mlp_dim, activation='gelu')(representation)
        features = layers.Dropout(dropout_rate)(features)
        outputs = layers.Dense(self.num_classes, activation='softmax')(features)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("\n✓ Vision Transformer built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        print(f"✓ Number of patches: {self.num_patches}")
        print(f"✓ Transformer layers: {transformer_layers}")
        
        # Save architecture
        with open(self.model_dir / 'model_architecture.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        return model
    
    def _create_patches(self, images):
        """Extract patches from images"""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches
    
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
        """Plot training history"""
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        print("\n📈 Generating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title(f'{self.model_name} - Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
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
