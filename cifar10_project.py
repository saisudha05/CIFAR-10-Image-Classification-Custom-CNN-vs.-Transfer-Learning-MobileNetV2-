"""
CIFAR-10 Deep Learning Project
================================
Approach:
  1. Custom CNN trained from scratch (primary model)
  2. Transfer Learning via MobileNetV2 (feature extractor + custom head)
  3. Side-by-side evaluation and comparison

Dataset: CIFAR-10 (.mat format), 10 classes, 60,000 images (50k train / 10k test)
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── TF / Keras ──────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"TensorFlow {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU') or 'None (CPU mode)'}")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/claude/cifar-10-batches-mat"
OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
IMG_SIZE    = 32
NUM_CLASSES = 10
BATCH_SIZE  = 64
EPOCHS_CNN  = 30   # custom CNN
EPOCHS_TL   = 20   # transfer-learning head
SEED        = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_batch(path):
    d = sio.loadmat(path)
    imgs   = d['data'].astype(np.float32)          # (N, 3072)
    labels = d['labels'].flatten().astype(np.int32) # (N,)
    # CIFAR stores as 3×32×32 row-major; reshape → (N,3,32,32) → (N,32,32,3)
    imgs = imgs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return imgs, labels

print("\n── Loading data ──")
train_imgs, train_labels = [], []
for i in range(1, 6):
    imgs, labels = load_batch(f"{DATA_DIR}/data_batch_{i}.mat")
    train_imgs.append(imgs);  train_labels.append(labels)

X_train = np.concatenate(train_imgs)   # (50000, 32, 32, 3)
y_train = np.concatenate(train_labels) # (50000,)
X_test,  y_test  = load_batch(f"{DATA_DIR}/test_batch.mat")

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Normalise ─────────────────────────────────────────────────────────────
mean = X_train.mean(axis=(0,1,2), keepdims=True)
std  = X_train.std (axis=(0,1,2), keepdims=True) + 1e-7

X_train_n = (X_train - mean) / std
X_test_n  = (X_test  - mean) / std

# One-hot labels
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

# ── Validation split (10 %) ────────────────────────────────────────────────
val_size  = 5000
X_val, y_val_oh = X_train_n[-val_size:], y_train_oh[-val_size:]
X_tr,  y_tr_oh  = X_train_n[:-val_size], y_train_oh[:-val_size]
print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_test_n.shape}")

# ── Augmentation ──────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
datagen.fit(X_tr)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL A – Custom CNN from Scratch
# ─────────────────────────────────────────────────────────────────────────────
def build_custom_cnn(input_shape=(32,32,3), num_classes=10):
    """
    A moderately deep CNN with:
      • 3 Conv blocks (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout)
      • Residual-style skip connection in block 3
      • Global Average Pooling + dense head
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3 with skip connection
    shortcut = layers.Conv2D(256, 1, padding='same')(x)   # 1×1 projection
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='CustomCNN')

cnn_model = build_custom_cnn()
cnn_model.summary()

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_cnn = [
    EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint(f"{OUT_DIR}/custom_cnn_best.keras",
                    save_best_only=True, monitor='val_accuracy', verbose=1),
]

print("\n── Training Custom CNN ──")
history_cnn = cnn_model.fit(
    datagen.flow(X_tr, y_tr_oh, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_oh),
    epochs=EPOCHS_CNN,
    callbacks=callbacks_cnn,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL B – Transfer Learning (MobileNetV2 feature extractor)
# ─────────────────────────────────────────────────────────────────────────────
def build_transfer_model(num_classes=10):
    """
    MobileNetV2 pre-trained on ImageNet, weights frozen, custom classification
    head trained from scratch.  Images are upscaled 32→96 so MobileNetV2 gets
    a reasonable input size.
    """
    inp = keras.Input(shape=(32, 32, 3))
    # Upsample to 96×96 (MobileNetV2 min recommended: 96)
    x = layers.UpSampling2D(size=(3, 3), interpolation='bilinear')(inp)
    # Rescale to [0,1] then to MobileNetV2 [-1,1]
    # (X_train_n is already z-normalised; we invert roughly then apply preprocess)
    x = layers.Lambda(lambda t: t * std.squeeze() + mean.squeeze())(x)
    x = layers.Rescaling(1./255)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x * 255)   # expects [0,255]

    base = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False   # freeze all base weights

    features = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inp, out, name='TransferLearning_MobileNetV2')
    return model, base

tl_model, base_model = build_transfer_model()
tl_model.summary()

tl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_tl = [
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(f"{OUT_DIR}/transfer_learning_best.keras",
                    save_best_only=True, monitor='val_accuracy', verbose=1),
]

print("\n── Training Transfer Learning Model (frozen base) ──")
history_tl = tl_model.fit(
    datagen.flow(X_tr, y_tr_oh, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_oh),
    epochs=EPOCHS_TL,
    callbacks=callbacks_tl,
    verbose=1,
)

# ── Fine-tune top layers of MobileNetV2 ──────────────────────────────────────
print("\n── Fine-tuning top 30 layers of MobileNetV2 ──")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

tl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_tl_ft = tl_model.fit(
    datagen.flow(X_tr, y_tr_oh, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_oh),
    epochs=10,
    callbacks=callbacks_tl,
    verbose=1,
)

# Merge histories
for k in history_tl.history:
    history_tl.history[k] += history_tl_ft.history.get(k, [])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test Evaluation ──")
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_n, y_test_oh, verbose=0)
tl_loss,  tl_acc  = tl_model.evaluate (X_test_n, y_test_oh, verbose=0)
print(f"Custom CNN        – loss: {cnn_loss:.4f}  accuracy: {cnn_acc*100:.2f}%")
print(f"Transfer Learning – loss: {tl_loss:.4f}  accuracy: {tl_acc*100:.2f}%")

cnn_preds = np.argmax(cnn_model.predict(X_test_n, verbose=0), axis=1)
tl_preds  = np.argmax(tl_model .predict(X_test_n, verbose=0), axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['accuracy'],     label='Train Acc')
    ax1.plot(history['val_accuracy'], label='Val Acc')
    ax1.set_title(f'{title} – Accuracy'); ax1.legend(); ax1.set_xlabel('Epoch')
    ax2.plot(history['loss'],     label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title(f'{title} – Loss'); ax2.legend(); ax2.set_xlabel('Epoch')
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"Saved: {path}")

plot_training(history_cnn.history, 'Custom CNN',
              f"{OUT_DIR}/cnn_training_curves.png")
plot_training(history_tl.history,  'Transfer Learning (MobileNetV2)',
              f"{OUT_DIR}/tl_training_curves.png")

def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"Saved: {path}")

plot_confusion_matrix(y_test, cnn_preds, 'Custom CNN – Confusion Matrix',
                      f"{OUT_DIR}/cnn_confusion_matrix.png")
plot_confusion_matrix(y_test, tl_preds,  'Transfer Learning – Confusion Matrix',
                      f"{OUT_DIR}/tl_confusion_matrix.png")

# Comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
models = ['Custom CNN', 'Transfer Learning\n(MobileNetV2)']
accs   = [cnn_acc * 100, tl_acc * 100]
bars   = ax.bar(models, accs, color=['steelblue', 'darkorange'], width=0.4)
ax.set_ylim(0, 100)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Model Comparison – CIFAR-10 Test Accuracy')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=120)
plt.close()
print(f"Saved: {OUT_DIR}/model_comparison.png")

# Sample predictions grid
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), 32, replace=False)
for ax, i in zip(axes.flatten(), idx):
    img = (X_test_n[i] * std.squeeze() + mean.squeeze()).clip(0, 255).astype(np.uint8)
    ax.imshow(img)
    true_lbl = CLASS_NAMES[y_test[i]]
    pred_lbl = CLASS_NAMES[cnn_preds[i]]
    color = 'green' if y_test[i] == cnn_preds[i] else 'red'
    ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", fontsize=6, color=color)
    ax.axis('off')
plt.suptitle('Custom CNN – Sample Predictions (green=correct, red=wrong)', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sample_predictions.png", dpi=120)
plt.close()
print(f"Saved: {OUT_DIR}/sample_predictions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLASSIFICATION REPORTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Custom CNN Classification Report ──")
print(classification_report(y_test, cnn_preds, target_names=CLASS_NAMES))

print("\n── Transfer Learning Classification Report ──")
print(classification_report(y_test, tl_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE FINAL MODELS
# ─────────────────────────────────────────────────────────────────────────────
cnn_model.save(f"{OUT_DIR}/custom_cnn_final.keras")
tl_model .save(f"{OUT_DIR}/transfer_learning_final.keras")
print(f"\nModels saved to {OUT_DIR}/")
print("\n✅  Done!")
