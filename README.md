# CIFAR-10-Image-Classification-Custom-CNN-vs.-Transfer-Learning-MobileNetV2-

A comparative deep learning project evaluating a CNN built from scratch against a fine-tuned MobileNetV2 transfer-learning model on the CIFAR-10 image classification benchmark.
Project Overview

CIFAR-10 is a benchmark dataset of 60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into 50,000 training and 10,000 test images. The low resolution makes fine-grained classification genuinely difficult — even for humans.

**Objectives:**
Classify CIFAR-10 images across all 10 categories using deep learning
Compare a custom CNN trained from scratch against transfer learning with MobileNetV2
Evaluate accuracy, precision, recall, and F1-score per class
Analyze strengths, weaknesses, and failure modes of each approach

**Tools & Technologies**
Python
TensorFlow,Keras
NumPy,Pandas
MatPlotlib,Seaborn
Scikit-Learn
Jupyter Notebook

**Methodology**

1. Data Preprocessing — Pixel normalization, image resizing, and stratified split into 45,000 train / 5,000 validation / 10,000 test images. Data augmentation (flips, rotations, shifts, zoom) was applied during training.

2. Custom CNN — Built from scratch: 3 convolutional blocks (64 → 128 → 256 filters), a residual skip connection, and a 512-unit dense head (~1.3M trainable parameters). Trained for 10 epochs (batch size 128) with Adam, ReduceLROnPlateau, ModelCheckpoint, and Batch Normalization + Dropout for regularization.

3. MobileNetV2 Transfer Learning — Pretrained ImageNet backbone with a custom dense classification head, trained in two phases: (1) frozen-base feature extraction, (2) fine-tuning with the top 30 layers unfrozen at a reduced learning rate (1e-5).

4. Evaluation — Accuracy, precision, recall, and F1-score per class on the held-out test set, plus confusion matrix and sample-prediction analysis for both models.


Key finding: The custom CNN outperformed the MobileNetV2 transfer-learning model on this task, likely because it learned features directly from the native 32×32 resolution, whereas MobileNetV2's ImageNet-pretrained features are optimized for much higher-resolution inputs. Interestingly, MobileNetV2 held its own (or won outright) on dog and deer — the two classes both models struggled with most — suggesting its pretrained features generalize slightly better on the hardest, most visually ambiguous categories.

Both models showed a similar failure pattern: confusion was concentrated among visually similar animal classes (dog ↔ cat, deer ↔ frog, bird ↔ deer) rather than spread randomly — evidence that both learned genuine, structured class features rather than noise.

**Future Work**


Hyperparameter optimization and expanded data augmentation
Deeper / alternative custom CNN architectures
Exploring EfficientNet and ResNet as transfer-learning backbones
Training and evaluating on higher-resolution datasets
