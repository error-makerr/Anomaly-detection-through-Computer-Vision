## Industrial Defect Detection Using Hybrid Deep Learning
This repository contains the implementation of my thesis project: Computer Vision Based Deep Learning Approaches for Automated Visual Inspection and Defect Detection in Industrial Environments.

The project explores supervised, unsupervised, and self-supervised learning paradigms across 16 models, and introduces a novel hybrid fusion framework that combines the strengths of MobileNetV2 and DINO. The system is designed to:

Achieve state-of-the-art accuracy on known defect types

Detect unknown defects through anomaly detection

Provide explainable predictions with attention-based heatmaps

Run efficiently on edge devices (e.g., Raspberry Pi, Jetson Nano)

This work contributes a deployment-ready solution for industrial quality assurance, bridging academic research and real-world manufacturing needs.


✨ Features
Hybrid Fusion Architecture  
Combines MobileNetV2 (supervised) and DINO (self-supervised) for robust defect detection across industrial datasets.

Explainable Predictions  
Attention maps and Grad-CAM visualizations highlight defect regions, making the system interpretable and trustworthy.

Unknown Defect Detection  
Capable of flagging anomalies not seen during training, ensuring adaptability to evolving industrial environments.

Comprehensive Benchmarking  
Evaluates 16 models across supervised, unsupervised, and self-supervised paradigms on a unified multi-domain dataset.

Deployment Ready  
Optimized for edge devices (Raspberry Pi, Jetson Nano) with lightweight inference and high accuracy.

Ablation Studies  
Demonstrates the necessity of each branch (MobileNet, DINO, Fusion) through systematic component analysis.

Cross-Domain Generalization  
Trained and tested on merged datasets (MVTec AD, Casting Product, Magnetic Tile) to ensure robustness across diverse defect types.
