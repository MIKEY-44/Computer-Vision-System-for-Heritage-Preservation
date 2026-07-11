"""
Pushkarani Temple Pond Classification - Multi-Model Training Script
Each model gets its own folder with all outputs
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, InceptionV3, DenseNet121 ,MobileNetV3Small
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
EPOCHS = 100
DATASET_PATH = 'dataset'
TRAIN_SPLIT = 0.75

# ========================================
# CHOOSE YOUR MODEL NAME HERE
# ========================================
MODEL_NAME = "MobileNetV3Small"  
# Available: mobilenet, resnet50, vgg16, inception, densenet, custom
# Or use your own name like: "model_v1", "experiment_1", etc.
# ========================================

def split_dataset():
    """Split dataset into train and test folders"""
    
    print("Splitting dataset into train and test...")
    
    train_dir = 'dataset_split/train'
    test_dir = 'dataset_split/test'
    
    if os.path.exists('dataset_split'):
        shutil.rmtree('dataset_split')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(DATASET_PATH) 
               if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    print(f"Found classes: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * TRAIN_SPLIT)
        
        train_images = image_files[:split_idx]
        test_images = image_files[split_idx:]
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
        
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)
        
        print(f"  {class_name}: {len(train_images)} train, {len(test_images)} test")
    
    print("\nDataset split completed!")
    return train_dir, test_dir

def create_data_generators(train_dir, test_dir):
    """Create data generators with HEAVY augmentation for small dataset"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        channel_shift_range=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator

def create_custom_model(num_classes=3):
    """Custom CNN model with heavy regularization"""
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_model(model_name, num_classes=3):
    """Create model based on model_name"""
    
    model_type = model_name.lower()
    base_model = None
    
    print(f"\n{'='*60}")
    print(f"Creating Model: {model_name.upper()}")
    print(f"{'='*60}")
    
    if model_type == "custom":
        print("Building custom CNN model...")
        model = create_custom_model(num_classes)
        return model, None
    
    # Transfer Learning Models
    try:
        if model_type == "mobilenet":
            print("Loading MobileNetV2...")
            base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        elif model_type == "resnet50":
            print("Loading ResNet50...")
            base_model = ResNet50(weights='imagenet', include_top=False,
                                 input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        elif model_type == "vgg16":
            print("Loading VGG16...")
            base_model = VGG16(weights='imagenet', include_top=False,
                              input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        elif model_type == "inception" or model_type == "inceptionv3":
            print("Loading InceptionV3...")
            base_model = InceptionV3(weights='imagenet', include_top=False,
                                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        elif model_type == "densenet":
            print("Loading DenseNet121...")
            base_model = DenseNet121(weights='imagenet', include_top=False,
                                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        else:
            print(f"Model type '{model_type}' not recognized as a pre-trained model.")
            print("Falling back to MobileNetV2...")
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        print(f"✓ Base model loaded successfully!")
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model on top
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model, base_model
        
    except Exception as e:
        print(f"Error loading transfer learning model: {e}")
        print("Falling back to custom CNN...")
        model = create_custom_model(num_classes)
        return model, None

def plot_training_history(history, output_dir):
    """Plot training and testing accuracy/loss"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Testing Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Testing Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'training_history.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training history saved as '{filename}'")
    plt.show()

def evaluate_model(model, test_generator, output_dir):
    """Evaluate model on test set and create confusion matrix"""
    
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print("=" * 60)
    report = classification_report(true_classes, predicted_classes, 
                                   target_names=class_labels, 
                                   output_dict=True,
                                   zero_division=0)
    print(classification_report(true_classes, predicted_classes, 
                               target_names=class_labels,
                               zero_division=0))
    
    print("Per-Class Performance:")
    print("-" * 60)
    for class_name in class_labels:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        print(f"{class_name}:")
        print(f"  Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | "
              f"F1-Score: {f1*100:.2f}% | Images: {int(support)}")
    
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\nAccuracy: {test_accuracy*100:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved as '{filename}'")
    plt.show()
    
    return test_accuracy, test_loss, report

def main():
    """Main training function"""
    
    print("=" * 60)
    print("Pushkarani Classification - Multi-Model Trainer")
    print("=" * 60)
    print(f"\nModel Name: {MODEL_NAME}")
    print("=" * 60)
    
    # Create output folder for this model
    output_dir = MODEL_NAME
    if os.path.exists(output_dir):
        print(f"\n⚠️  Warning: Folder '{output_dir}' already exists!")
        response = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print("Training cancelled. Please change MODEL_NAME.")
            return
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Created output folder: {output_dir}/")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found!")
        return
    
    # Split dataset
    print("\n1. Splitting dataset...")
    print(f"   Split ratio: {TRAIN_SPLIT*100:.0f}% train, {(1-TRAIN_SPLIT)*100:.0f}% test")
    train_dir, test_dir = split_dataset()
    
    # Create data generators
    print("\n2. Creating data generators with heavy augmentation...")
    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    
    print(f"\nDataset Information:")
    print(f"  - Training images: {train_generator.samples}")
    print(f"  - Testing images: {test_generator.samples}")
    print(f"  - Classes: {list(train_generator.class_indices.keys())}")
    
    # Create model
    print("\n3. Creating model...")
    model, base_model = create_model(MODEL_NAME, num_classes=len(train_generator.class_indices))
    
    use_transfer_learning = (base_model is not None)
    
    # Compile model
    print("\n4. Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # File paths inside model folder
    best_model_phase1 = os.path.join(output_dir, 'best_model_phase1.keras')
    best_model_final = os.path.join(output_dir, 'best_model.keras')
    final_model = os.path.join(output_dir, 'final_model.keras')
    
    # Callbacks for phase 1
    callbacks_phase1 = [
        ModelCheckpoint(
            best_model_phase1,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train Phase 1
    print("\n" + "=" * 60)
    if use_transfer_learning:
        print("Starting Phase 1: Training Top Layers Only...")
    else:
        print("Starting Training with Custom CNN...")
    print("=" * 60)
    
    history1 = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=50,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tune if using transfer learning
    if use_transfer_learning:
        print("\n5. Starting Phase 2: Fine-tuning base model...")
        
        base_model.trainable = True
        unfrozen_layers = min(30, len(base_model.layers))
        for layer in base_model.layers[:-unfrozen_layers]:
            layer.trainable = False
        
        print(f"   Unfreezing last {unfrozen_layers} layers of base model")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks_phase2 = [
            ModelCheckpoint(
                best_model_final,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        print("\n" + "=" * 60)
        print("Starting Phase 2: Fine-tuning...")
        print("=" * 60)
        
        history2 = model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=50,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        history_combined = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
    else:
        if os.path.exists(best_model_phase1):
            shutil.copy2(best_model_phase1, best_model_final)
        
        history_combined = {
            'accuracy': history1.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'],
            'loss': history1.history['loss'],
            'val_loss': history1.history['val_loss']
        }
    
    class History:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history = History(history_combined)
    
    # Save final model
    model.save(final_model)
    print(f"\nFinal model saved as '{final_model}'")
    
    # Plot training history
    print("\n6. Generating visualizations...")
    plot_training_history(combined_history, output_dir)
    
    # Evaluate model
    print("\n7. Evaluating model on test set...")
    test_accuracy, test_loss, report = evaluate_model(model, test_generator, output_dir)
    
    # Save class indices
    import json
    class_indices = train_generator.class_indices
    class_indices_file = os.path.join(output_dir, 'class_indices.json')
    with open(class_indices_file, 'w') as f:
        json.dump(class_indices, f, indent=4)
    print(f"\nClass indices saved as '{class_indices_file}'")
    
    # Save training summary
    summary = {
        'model_name': MODEL_NAME,
        'dataset_path': DATASET_PATH,
        'train_samples': train_generator.samples,
        'test_samples': test_generator.samples,
        'train_split': TRAIN_SPLIT,
        'classes': list(class_indices.keys()),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'total_epochs': len(history_combined['loss']),
        'final_train_accuracy': float(history_combined['accuracy'][-1]),
        'final_test_accuracy': float(history_combined['val_accuracy'][-1]),
        'used_transfer_learning': use_transfer_learning,
        'per_class_metrics': {
            cls: {
                'precision': float(report[cls]['precision']),
                'recall': float(report[cls]['recall']),
                'f1_score': float(report[cls]['f1-score']),
                'support': int(report[cls]['support'])
            }
            for cls in list(class_indices.keys())
        }
    }
    
    summary_file = os.path.join(output_dir, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Training summary saved as '{summary_file}'")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModel Folder: {output_dir}/")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print("\nAll files saved in folder structure:")
    print(f"{output_dir}/")
    print(f"  ├── best_model.keras")
    print(f"  ├── final_model.keras")
    print(f"  ├── class_indices.json")
    print(f"  ├── training_history.png")
    print(f"  ├── confusion_matrix.png")
    print(f"  └── training_summary.json")
    
    if use_transfer_learning:
        print(f"\n✓ Used Transfer Learning")
    else:
        print("\n✓ Used Custom CNN")
    
    print("\nCleaning up temporary directories...")
    shutil.rmtree('dataset_split')
    
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS:")
    print("  - mobilenet    (Fast, accurate)")
    print("  - resnet50     (Deep, powerful)")
    print("  - vgg16        (Classic, reliable)")
    print("  - inception    (Multi-scale features)")
    print("  - densenet     (Dense connections)")
    print("  - custom       (Custom CNN)")
    print("\nOr use any custom name like: model_v1, exp_1, etc.")
    print("=" * 60)

if __name__ == "__main__":
    main()