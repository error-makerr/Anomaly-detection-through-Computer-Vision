"""Data Augmentation Utilities for Defect Detection"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import csv
from pathlib import Path


class MixupCutmixAugmenter:
    """Mixup and CutMix data augmentation strategies"""
    
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixing ratio
        """
        self.alpha = alpha
    
    def mixup(self, x1, y1, x2, y2):
        """
        Mixup augmentation: blend two images and their labels
        
        Args:
            x1: First image (batch or single)
            y1: First label
            x2: Second image (batch or single)
            y2: Second label
        
        Returns:
            Mixed image and blended label
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        
        return x, y
    
    def cutmix(self, x1, y1, x2, y2):
        """
        CutMix augmentation: cut and paste regions between two images
        
        Args:
            x1: First image (batch or single)
            y1: First label
            x2: Second image (batch or single)
            y2: Second label
        
        Returns:
            CutMixed image and blended label
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        h, w = x1.shape[0], x1.shape[1]
        
        # Random cut size
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random cut position
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        x = x1.copy()
        x[bby1:bby2, bbx1:bbx2] = x2[bby1:bby2, bbx1:bbx2]
        
        # Recalculate lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (h * w)
        
        y = lam * y1 + (1 - lam) * y2
        
        return x, y
    
    def augment_batch(self, x, y, augmentation_type='mixup'):
        """
        Apply augmentation to a batch of images
        
        Args:
            x: Batch of images
            y: Batch of labels
            augmentation_type: 'mixup' or 'cutmix'
        
        Returns:
            Augmented batch of images and labels
        """
        batch_size = x.shape[0]
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        
        x_aug = x.copy()
        y_aug = y.copy()
        
        for i in range(batch_size // 2):
            idx1 = indices[2*i]
            idx2 = indices[2*i + 1]
            
            if augmentation_type == 'mixup':
                x_aug[idx1], y_aug[idx1] = self.mixup(
                    x[idx1], y[idx1], 
                    x[idx2], y[idx2]
                )
            elif augmentation_type == 'cutmix':
                x_aug[idx1], y_aug[idx1] = self.cutmix(
                    x[idx1], y[idx1], 
                    x[idx2], y[idx2]
                )
        
        return x_aug, y_aug


class SimpleCSVLogger(Callback):
    """Simple CSV logger callback for training metrics"""
    
    def __init__(self, filename='training_log.csv', append=False):
        """
        Args:
            filename: Path to CSV file
            append: Whether to append to existing file
        """
        super(SimpleCSVLogger, self).__init__()
        self.filename = filename
        self.append = append
        self.csv_file = None
        self.writer = None
        self.epochs_written = 0
        self.keys = None
    
    def on_train_begin(self, logs=None):
        """Called at the start of training"""
        logs = logs or {}
        
        # Create output directory if needed
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        
        self.keys = sorted(logs.keys())
        
        # Check if file exists
        file_exists = Path(self.filename).exists()
        
        if file_exists and self.append:
            self.csv_file = open(self.filename, 'a', newline='')
        else:
            self.csv_file = open(self.filename, 'w', newline='')
        
        self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + self.keys)
        
        # Write header only if file is new
        if not file_exists or not self.append:
            self.writer.writeheader()
        
        self.csv_file.flush()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        logs = logs or {}
        
        # Filter logs to only include the keys we care about
        row_dict = {k: v for k, v in logs.items() if k in self.keys}
        row_dict['epoch'] = epoch
        
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        if self.csv_file is not None:
            self.csv_file.close()


class AugmentationStrategy:
    """Base class for augmentation strategies"""
    
    def __init__(self, prob=0.5):
        """
        Args:
            prob: Probability of applying augmentation
        """
        self.prob = prob
    
    def __call__(self, image):
        """Apply augmentation with probability"""
        if np.random.random() < self.prob:
            return self.apply(image)
        return image
    
    def apply(self, image):
        """Override in subclasses"""
        raise NotImplementedError


class RandomBrightnessContrast(AugmentationStrategy):
    """Random brightness and contrast adjustment"""
    
    def __init__(self, brightness_delta=0.3, contrast_range=(0.7, 1.3), prob=0.5):
        super().__init__(prob)
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
    
    def apply(self, image):
        """Apply random brightness and contrast"""
        # Random brightness
        if np.random.random() < 0.5:
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            image = tf.image.adjust_brightness(image, delta)
        
        # Random contrast
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
            image = tf.image.adjust_contrast(image, contrast_factor)
        
        return image


class RandomRotation(AugmentationStrategy):
    """Random rotation augmentation"""
    
    def __init__(self, max_angle=30, prob=0.5):
        super().__init__(prob)
        self.max_angle = max_angle
    
    def apply(self, image):
        """Apply random rotation"""
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        # Convert angle to radians
        angle_rad = angle * np.pi / 180
        return tf.raw_ops.RotateImage(images=tf.expand_dims(image, 0), 
                                      angles=[angle_rad])[0]


class RandomSaturation(AugmentationStrategy):
    """Random saturation adjustment"""
    
    def __init__(self, saturation_range=(0.7, 1.3), prob=0.5):
        super().__init__(prob)
        self.saturation_range = saturation_range
    
    def apply(self, image):
        """Apply random saturation adjustment"""
        saturation_factor = np.random.uniform(self.saturation_range[0], 
                                              self.saturation_range[1])
        image = tf.image.adjust_saturation(image, saturation_factor)
        return image


class AugmentationPipeline:
    """Compose multiple augmentation strategies"""
    
    def __init__(self, augmentations=None):
        """
        Args:
            augmentations: List of augmentation strategies
        """
        self.augmentations = augmentations or []
    
    def add(self, augmentation):
        """Add an augmentation to the pipeline"""
        self.augmentations.append(augmentation)
        return self
    
    def __call__(self, image):
        """Apply all augmentations in sequence"""
        for aug in self.augmentations:
            image = aug(image)
        return image
