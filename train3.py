"""
Pushkarani Stone Quality & Maintainability Classification - Advanced Training Script
Trains on: stabDataset/Good, stabDataset/Medium, stabDataset/Bad
Focuses on: Stone condition, structural integrity, maintenance status
Preprocessing: Removes water, unwanted objects - Extracts ONLY stone regions
Uses: DenseNet121 + Custom preprocessing for stone surface analysis
Enhanced Accuracy: Image processing + Edge detection + Color-based filtering
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import shutil
import json
from datetime import datetime
import cv2
from PIL import Image
import io

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8   # Smaller batch for better quality
EPOCHS = 200     # More epochs for convergence
DATASET_PATH = 'stabDataset'
TRAIN_SPLIT = 0.8
MODEL_OUTPUT_DIR = 'stone_quality_model'
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

CLASSES = ['Good', 'Bad', 'Medium']
NUM_CLASSES = 3

print(f"[INFO] Stone Quality Classification (Advanced Preprocessing)")
print(f"  - Dataset Path: {DATASET_PATH}")
print(f"  - Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"  - Model: DenseNet121 (Best accuracy)")
print(f"  - Preprocessing: Water removal + Stone extraction")
print(f"  - Classes: {CLASSES}")
# ======================================================

# ==================== IMAGE PREPROCESSING ====================

def extract_stone_region(image_path, target_size=(224, 224)):
    """
    Extract stone region from image by removing water and unwanted objects
    Uses: Color detection, edge detection, morphological operations
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        original_h, original_w = img.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask to remove water (blue/cyan colors)
        # Water is typically in the blue range
        lower_water = np.array([85, 40, 40])
        upper_water = np.array([130, 255, 255])
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Invert to keep stone (non-water)
        stone_mask = cv2.bitwise_not(water_mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, kernel)
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(stone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # Fallback: use entire image if no contours found
            processed = cv2.resize(img, target_size)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            return processed
        
        # Get largest contour (main stone structure)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original_w - x, w + 2 * padding)
        h = min(original_h - y, h + 2 * padding)
        
        # Extract stone region
        stone_region = img[y:y+h, x:x+w]
        
        # Resize to target size
        processed = cv2.resize(stone_region, target_size)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        return processed
        
    except Exception as e:
        print(f"    Warning: Preprocessing failed for {image_path}, using original")
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                img = cv2.resize(img, target_size)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        return None

def enhance_stone_features(image):
    """Enhance stone texture features using CLAHE"""
    try:
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    except:
        return image

def preprocess_image_with_extraction(image_data):
    """Preprocess image: extract stone region + enhance"""
    try:
        if isinstance(image_data, str):
            # File path
            processed = extract_stone_region(image_data)
        else:
            # File object or PIL Image
            if hasattr(image_data, 'name'):
                processed = extract_stone_region(image_data.name)
            else:
                img = Image.open(image_data).convert('RGB')
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                processed = extract_stone_region(img_cv)
        
        if processed is None:
            return None
        
        # Enhance features
        enhanced = enhance_stone_features(processed)
        
        # Normalize
        normalized = enhanced.astype(np.float32) / 255.0
        normalized = np.expand_dims(normalized, axis=0)
        
        return normalized
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def verify_dataset():
    """Verify stabDataset structure and count images"""
    print("\n[*] Verifying stabDataset structure...")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path '{DATASET_PATH}' not found!")
    
    class_counts = {}
    total_images = 0
    
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Class folder '{class_name}' not found in {DATASET_PATH}")
        
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        class_counts[class_name] = len(images)
        total_images += len(images)
        print(f"  ✓ {class_name:8} - {len(images):4} images")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {class_name} folder!")
    
    print(f"  ✓ Total images: {total_images}")
    return class_counts

def split_dataset():
    """Split stabDataset into train and validation with stone extraction"""
    print("\n[*] Splitting dataset and extracting stone regions...")
    
    train_dir = os.path.join(MODEL_OUTPUT_DIR, 'data', 'train')
    val_dir = os.path.join(MODEL_OUTPUT_DIR, 'data', 'validation')
    
    if os.path.exists(os.path.join(MODEL_OUTPUT_DIR, 'data')):
        shutil.rmtree(os.path.join(MODEL_OUTPUT_DIR, 'data'))
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    split_stats = {}
    
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * TRAIN_SPLIT)
        
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        print(f"\n  Processing {class_name}...")
        
        # Process and copy training images
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            
            try:
                extracted = extract_stone_region(src)
                if extracted is not None:
                    # Save extracted stone region
                    pil_img = Image.fromarray((extracted * 255).astype(np.uint8))
                    pil_img.save(dst)
                else:
                    shutil.copy2(src, dst)
            except:
                shutil.copy2(src, dst)
        
        # Process and copy validation images
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            
            try:
                extracted = extract_stone_region(src)
                if extracted is not None:
                    pil_img = Image.fromarray((extracted * 255).astype(np.uint8))
                    pil_img.save(dst)
                else:
                    shutil.copy2(src, dst)
            except:
                shutil.copy2(src, dst)
        
        split_stats[class_name] = {
            'train': len(train_images),
            'validation': len(val_images),
            'total': len(image_files)
        }
        
        print(f"    ✓ Train: {len(train_images)}, Val: {len(val_images)}")
    
    return train_dir, val_dir, split_stats

def create_data_generators(train_dir, val_dir):
    """Create data generators with HEAVY augmentation and preprocessing for stone surface analysis"""
    
    print("\n[*] Creating data generators with stone-focused augmentation and preprocessing...")
    
    def preprocess_with_enhancement(img):
        """Apply CLAHE enhancement to batch images"""
        if isinstance(img, np.ndarray):
            # Convert to float if needed
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            
            # Ensure RGB format
            if len(img.shape) == 2:
                img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            elif img.shape[2] == 4:
                img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGBA2RGB).astype(np.float32) / 255.0
            
            # Apply CLAHE enhancement
            img_enhanced = enhance_stone_features(img)
            return img_enhanced
        return img
    
    # Training augmentation - AGGRESSIVE for stone texture recognition
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_with_enhancement,
        rotation_range=60,              # Stone weathering from different angles
        width_shift_range=0.4,          # Cracks and damage patterns
        height_shift_range=0.4,
        shear_range=0.4,                # Structural deformations
        zoom_range=0.4,                 # Close-up stone details
        horizontal_flip=True,            # Mirror symmetry
        vertical_flip=True,              # Vertical variations
        brightness_range=[0.6, 1.4],    # Lighting conditions (sun/shadow)
        fill_mode='nearest',
        channel_shift_range=25.0,       # Color variations in stone
    )
    
    # Validation - preprocessing only, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_with_enhancement
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    print(f"  ✓ Training samples: {train_generator.samples}")
    print(f"  ✓ Validation samples: {val_generator.samples}")
    print(f"  ✓ Class mapping: {train_generator.class_indices}")
    
    return train_generator, val_generator, train_generator.class_indices

def build_efficientnetv2_model(num_classes):
    """Build DenseNet121 model - Best for stone texture analysis with preprocessing"""
    
    print("\n[*] Building DenseNet121 model...")
    
    # Load pretrained DenseNet121
    base_model = DenseNet121(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build custom top with more regularization
    model = keras.Sequential([
        base_model,
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE, decay=0.00001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  ✓ Model built with DenseNet121")
    print(f"  ✓ Total parameters: {model.count_params():,}")
    
    return model, base_model

def train_model(model, base_model, train_gen, val_gen):
    """Train model with progressive unfreezing strategy"""
    
    print("\n[*] Starting training with progressive unfreezing...")
    
    # Create output directories
    os.makedirs(os.path.join(MODEL_OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_OUTPUT_DIR, 'logs'), exist_ok=True)
    
    # Phase 1: Train only top layers (frozen base)
    print("\n  [Phase 1] Training top layers (base model frozen)...")
    
    callbacks_phase1 = [
        ModelCheckpoint(
            os.path.join(MODEL_OUTPUT_DIR, 'checkpoints', 'best_model_phase1.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(log_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs', 'phase1'))
    ]
    
    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen base
    print("\n  [Phase 2] Fine-tuning with unfrozen base model...")
    
    # Unfreeze last 40 layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    
    # Lower learning rate for fine-tuning
    optimizer = Adam(learning_rate=LEARNING_RATE / 10)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        ModelCheckpoint(
            os.path.join(MODEL_OUTPUT_DIR, 'checkpoints', 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(log_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs', 'phase2'))
    ]
    
    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=callbacks_phase2,
        initial_epoch=EPOCHS // 2,
        verbose=1
    )
    
    # Combine histories - handle different keys
    combined_history = {
        'loss': history_phase1.history.get('loss', []) + history_phase2.history.get('loss', []),
        'accuracy': history_phase1.history.get('accuracy', []) + history_phase2.history.get('accuracy', []),
        'val_loss': history_phase1.history.get('val_loss', []) + history_phase2.history.get('val_loss', []),
        'val_accuracy': history_phase1.history.get('val_accuracy', []) + history_phase2.history.get('val_accuracy', [])
    }
    
    # Clean up empty lists
    combined_history = {k: v if v else [0] for k, v in combined_history.items()}
    
    return combined_history, model

def evaluate_model(model, train_gen, val_gen, class_indices):
    """Evaluate model on validation set"""
    
    print("\n[*] Evaluating model on validation set...")
    
    # Get predictions
    val_steps = int(np.ceil(val_gen.samples / BATCH_SIZE))
    predictions = model.predict(val_gen, steps=val_steps)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, pred_classes)
    print(f"\n  ✓ Validation Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    class_names = list(class_indices.keys())
    print("\n[Classification Report]")
    print(classification_report(true_classes, pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    
    return accuracy, cm, true_classes, pred_classes

def plot_results(history, cm, true_classes, pred_classes, class_indices):
    """Plot training history and confusion matrix"""
    
    print("\n[*] Generating plots...")
    
    class_names = list(class_indices.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training history - Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training history - Loss
    axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0], cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Distribution of predictions vs actuals
    unique_classes = np.unique(true_classes)
    pred_dist = [np.sum(pred_classes == c) for c in unique_classes]
    true_dist = [np.sum(true_classes == c) for c in unique_classes]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, true_dist, width, label='Actual', alpha=0.8)
    axes[1, 1].bar(x + width/2, pred_dist, width, label='Predicted', alpha=0.8)
    axes[1, 1].set_title('Class Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'training_results.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to {os.path.join(MODEL_OUTPUT_DIR, 'training_results.png')}")
    plt.close()

def save_class_indices(class_indices):
    """Save class indices mapping"""
    
    indices_file = os.path.join(MODEL_OUTPUT_DIR, 'class_indices.json')
    with open(indices_file, 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    print(f"  ✓ Class indices saved to {indices_file}")

def save_training_summary(accuracy, class_counts, split_stats):
    """Save training summary"""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'EfficientNetV2B0',
        'dataset': 'stabDataset',
        'classes': CLASSES,
        'focus': 'Stone quality, structural integrity, weathering, maintainability',
        'image_size': f'{IMG_HEIGHT}x{IMG_WIDTH}',
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'validation_accuracy': float(accuracy),
        'validation_accuracy_percent': f'{accuracy*100:.2f}%',
        'class_distribution': class_counts,
        'train_test_split': split_stats,
        'model_path': os.path.join(MODEL_OUTPUT_DIR, 'checkpoints', 'best_model.keras'),
        'phase1_model_path': os.path.join(MODEL_OUTPUT_DIR, 'checkpoints', 'best_model_phase1.keras'),
        'architecture': {
            'base_model': 'EfficientNetV2B0',
            'include_top_layers': '512 -> 256 -> 128 -> 3 (classes)',
            'dropout_rates': '[0.4, 0.3, 0.2]',
            'batch_normalization': 'Yes',
            'regularization': 'L2 (weight decay 0.0001)'
        },
        'augmentation': {
            'rotation': 60,
            'shifts': 0.4,
            'zoom': 0.4,
            'brightness': '[0.6, 1.4]',
            'vertical_flip': True,
            'horizontal_flip': True,
            'focus': 'Stone texture, cracks, weathering patterns'
        },
        'training_strategy': 'Two-phase: frozen base (75 epochs) -> fine-tune (75 epochs)',
        'optimization': {
            'optimizer': 'Adam',
            'loss_function': 'categorical_crossentropy',
            'phase1_lr': LEARNING_RATE,
            'phase2_lr': f'{LEARNING_RATE/10}',
            'early_stopping_patience': '20 epochs'
        }
    }
    
    summary_file = os.path.join(MODEL_OUTPUT_DIR, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Training summary saved to {summary_file}")
    
    # Also save as human-readable text
    summary_txt = os.path.join(MODEL_OUTPUT_DIR, 'TRAINING_SUMMARY.txt')
    with open(summary_txt, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PUSHKARANI STONE QUALITY & MAINTAINABILITY - TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {summary['model']}\n")
        f.write(f"Dataset: {summary['dataset']}\n")
        f.write(f"Classes: {', '.join(CLASSES)}\n")
        f.write(f"Focus: {summary['focus']}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"  Validation Accuracy: {summary['validation_accuracy_percent']}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        for class_name, stats in split_stats.items():
            f.write(f"  {class_name.upper()}:\n")
            f.write(f"    - Total: {stats['total']} images\n")
            f.write(f"    - Train: {stats['train']} images\n")
            f.write(f"    - Validation: {stats['validation']} images\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"  Base: {summary['architecture']['base_model']}\n")
        f.write(f"  Top Layers: {summary['architecture']['include_top_layers']}\n")
        f.write(f"  Dropout: {summary['architecture']['dropout_rates']}\n")
        f.write(f"  Batch Norm: {summary['architecture']['batch_normalization']}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  Epochs: {summary['epochs']}\n")
        f.write(f"  Batch Size: {summary['batch_size']}\n")
        f.write(f"  Image Size: {summary['image_size']}\n\n")
        
        f.write("AUGMENTATION STRATEGY (Stone-Focused):\n")
        f.write(f"  Rotation: ±{summary['augmentation']['rotation']}°\n")
        f.write(f"  Shifts: ±{int(summary['augmentation']['shifts']*100)}%\n")
        f.write(f"  Zoom: ±{int(summary['augmentation']['zoom']*100)}%\n")
        f.write(f"  Brightness: {summary['augmentation']['brightness']}\n")
        f.write(f"  Vertical Flip: {summary['augmentation']['vertical_flip']}\n")
        f.write(f"  Horizontal Flip: {summary['augmentation']['horizontal_flip']}\n")
        f.write(f"  Focus: {summary['augmentation']['focus']}\n\n")
        
        f.write("MODEL PATHS:\n")
        f.write(f"  Best Model: {summary['model_path']}\n")
        f.write(f"  Phase 1 Model: {summary['phase1_model_path']}\n")
        f.write(f"  Output Directory: {MODEL_OUTPUT_DIR}\n")
    
    print(f"  ✓ Text summary saved to {summary_txt}")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("PUSHKARANI STONE QUALITY & MAINTAINABILITY - TRAINING")
    print("="*80)
    
    try:
        # Step 1: Verify dataset
        class_counts = verify_dataset()
        
        # Step 2: Split dataset
        train_dir, val_dir, split_stats = split_dataset()
        
        # Step 3: Create data generators
        train_gen, val_gen, class_indices = create_data_generators(train_dir, val_dir)
        
        # Step 4: Build model
        model, base_model = build_efficientnetv2_model(NUM_CLASSES)
        
        # Step 5: Train model (returns trained model)
        history, trained_model = train_model(model, base_model, train_gen, val_gen)
        
        # Step 6: Use the trained model from training
        model = trained_model
        
        # Step 7: Evaluate
        accuracy, cm, true_classes, pred_classes = evaluate_model(model, train_gen, val_gen, class_indices)
        
        # Step 8: Visualize results
        plot_results(history, cm, true_classes, pred_classes, class_indices)
        
        # Step 9: Save metadata
        save_class_indices(class_indices)
        save_training_summary(accuracy, class_counts, split_stats)
        
        # Step 10: Save final model
        final_model_path = os.path.join(MODEL_OUTPUT_DIR, 'final_model.keras')
        model.save(final_model_path)
        print(f"\n  ✓ Final model saved to {final_model_path}")
        
        print("\n" + "="*80)
        print(f"✓ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Model saved to: {MODEL_OUTPUT_DIR}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
