# train_85percent.py - Optimized para sa ~2,400 images (target 85%+ accuracy)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np

print(" OPTIMIZED CAT VS DOG CLASSIFIER (Target: 85%+)")
print("=" * 60)

# ==================== CONFIG ====================
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 15

# ==================== DATA PREPARATION ====================
print("\n Step 1: Loading dataset...")

base_dir = r'C:\Users\Wendel\Projects\Dog-Cat-identifier\dogcat-env\dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# AGGRESSIVE data augmentation (para kumalat ang limited data)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# ==================== BUILD MODEL ====================
print("\n Step 2: Building MobileNetV2 model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# I-freeze muna
base_model.trainable = False

# Magdagdag ng custom classifier (mas maraming layers para sa limited data)
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ==================== CALLBACKS ====================
print("\n Step 3: Setting up callbacks...")

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=0.00001,
    verbose=1,
    mode='min'
)

checkpoint = ModelCheckpoint(
    'best_model_85.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ==================== PHASE 1: FROZEN BASE ====================
print("\n🎯 Step 4: Training Phase 1 (Frozen Base)...")

history1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_PHASE1,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ==================== PHASE 2: FINE-TUNING ====================
print("\n Step 5: Training Phase 2 (Fine-tuning)...")

# I-unfreeze ang base model
base_model.trainable = True

# I-freeze ang unang maraming layers, i-train lang ang later layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Re-compile with lower learning rate
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history2 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_PHASE2,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ==================== EVALUATION ====================
print("\n Step 6: Evaluating model...")

# I-load ang best model
best_model = tf.keras.models.load_model('best_model_85.keras')

# Evaluate
val_loss, val_acc, val_precision, val_recall = best_model.evaluate(validation_generator)
print(f"\nFINAL RESULTS:")
print(f" Validation Accuracy: {val_acc:.2%}")
print(f" Precision: {val_precision:.2%}")
print(f" Recall: {val_recall:.2%}")

# ==================== SAVE FINAL MODEL ====================
best_model.save('cat_dog_classifier_85percent.keras')
print("\n Model saved as: cat_dog_classifier_85percent.keras")

# ==================== PLOT RESULTS ====================
# Pagsamahin ang history
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(1, len(acc) + 1)
phase1_epochs = len(history1.history['accuracy'])

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
plt.axvline(x=phase1_epochs, color='g', linestyle='--', linewidth=2, label='Fine-tuning Start')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Training Progress (Final: {val_acc[-1]:.1%})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.axvline(x=phase1_epochs, color='g', linestyle='--', linewidth=2, label='Fine-tuning Start')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle(f'Cat vs Dog Classifier - {val_acc[-1]:.1%} Validation Accuracy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_85percent.png', dpi=300)
plt.show()

print("\n" + "=" * 60)
print(f" TRAINING COMPLETE! Accuracy: {val_acc[-1]:.1%}")
print(" Model: cat_dog_classifier_85percent.keras")
print(" Graph: training_85percent.png")
print("=" * 60)
