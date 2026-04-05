"""
Plant Disease Detection - Model Training Script
Trains a CNN using MobileNetV2 transfer learning on PlantVillage dataset.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
CONFIG = {
    'dataset_path': 'dataset/PlantVillage',
    'model_save_path': 'model/plant_disease_model.h5',
    'history_save_path': 'model/training_history.json',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 0.0001,
    'val_split': 0.2,
    'seed': 42,
    'num_classes': 38,
}

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


# ─── DATA PREPARATION ────────────────────────────────────────────────────────
def create_data_generators():
    """Create training and validation data generators with augmentation."""
    print("\n[1/4] Setting up data generators...")
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        validation_split=CONFIG['val_split']
    )
    
    # Validation: only rescaling, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=CONFIG['val_split']
    )
    
    train_generator = train_datagen.flow_from_directory(
        CONFIG['dataset_path'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        seed=CONFIG['seed'],
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        CONFIG['dataset_path'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        seed=CONFIG['seed'],
        shuffle=False
    )
    
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Classes found: {len(train_generator.class_indices)}")
    
    return train_generator, val_generator


# ─── MODEL ARCHITECTURE ──────────────────────────────────────────────────────
def build_model(num_classes):
    """
    Build MobileNetV2-based transfer learning model.
    
    Architecture:
    - MobileNetV2 backbone (pretrained on ImageNet, top excluded)
    - Global Average Pooling
    - Batch Normalization + Dropout
    - Dense(512, relu)
    - Dropout(0.5)
    - Dense(num_classes, softmax)
    """
    print("\n[2/4] Building MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build custom classification head
    inputs = keras.Input(shape=(*CONFIG['img_size'], 3))
    
    # Data augmentation layer (GPU-accelerated)
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='PlantDiseaseDetector')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    print(f"  Base model: MobileNetV2 (frozen)")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Trainable parameters: {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    
    return model, base_model


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train_model(model, train_gen, val_gen):
    """Train the model in two phases: head only, then fine-tuning."""
    print("\n[3/4] Training model...")
    
    os.makedirs('model', exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            CONFIG['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train classification head only
    print("\n  Phase 1: Training classification head (10 epochs)...")
    history_phase1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_phase1


def fine_tune_model(model, base_model, train_gen, val_gen, initial_history):
    """Fine-tune the top layers of the base model."""
    print("\n  Phase 2: Fine-tuning (unfreeze top 30 layers)...")
    
    # Unfreeze top 30 layers of base model
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    callbacks = [
        ModelCheckpoint(
            CONFIG['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
    ]
    
    history_phase2 = model.fit(
        train_gen,
        epochs=CONFIG['epochs'] - 10,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    combined = {}
    for key in initial_history.history:
        combined[key] = (
            initial_history.history[key] + history_phase2.history.get(key, [])
        )
    
    return combined


# ─── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate_model(model, val_gen):
    """Evaluate model and generate metrics."""
    print("\n[4/4] Evaluating model...")
    
    # Get predictions
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    
    # Accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\n  Test Accuracy: {accuracy * 100:.2f}%")
    
    # Classification report
    class_labels = list(val_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    print("\n  Classification Report (sample):")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # Save report
    with open('model/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return y_true, y_pred, accuracy


# ─── PLOTTING ─────────────────────────────────────────────────────────────────
def plot_training_history(history):
    """Plot training/validation accuracy and loss curves."""
    print("\nGenerating training plots...")
    
    os.makedirs('model', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Plant Disease Detection - Training History', fontsize=16, fontweight='bold')
    
    epochs_range = range(1, len(history['accuracy']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs_range, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs_range, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.0])
    
    # Loss plot
    axes[1].plot(epochs_range, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model/training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5
    )
    plt.title('Confusion Matrix - Plant Disease Detection', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved: model/confusion_matrix.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   PLANT DISEASE DETECTION - MODEL TRAINING")
    print("   Dataset: PlantVillage | Architecture: MobileNetV2")
    print("=" * 60)
    
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow is required. Install: pip install tensorflow")
        return
    
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"\nERROR: Dataset not found at '{CONFIG['dataset_path']}'")
        print("\nTo get PlantVillage dataset:")
        print("  1. Download from: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("  2. Extract to: dataset/PlantVillage/")
        print("  3. Structure: dataset/PlantVillage/<ClassName>/<images>")
        return
    
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU available: {len(gpus)} device(s)")
    else:
        print("  Running on CPU (training will be slow)")
    
    # Run training pipeline
    train_gen, val_gen = create_data_generators()
    model, base_model = build_model(len(train_gen.class_indices))
    
    history_obj = train_model(model, train_gen, val_gen)
    history = fine_tune_model(model, base_model, train_gen, val_gen, history_obj)
    
    # Save history
    with open(CONFIG['history_save_path'], 'w') as f:
        json.dump(history, f)
    
    # Evaluate
    y_true, y_pred, accuracy = evaluate_model(model, val_gen)
    
    # Plot results
    plot_training_history(history)
    class_names = list(val_gen.class_indices.keys())
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Final Accuracy: {accuracy * 100:.2f}%")
    print(f"  Model saved: {CONFIG['model_save_path']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
