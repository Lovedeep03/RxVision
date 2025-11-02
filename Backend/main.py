import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetV2S
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# Configuration
class Config:
    BATCH_SIZE = 16  # Reduced batch size to improve generalization
    EPOCHS = 1
    EARLY_STOPPING_PATIENCE = 5  # Stop training if validation doesn't improve
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    LR = 5e-4  # Reduced learning rate to prevent quick overfitting
    WEIGHT_DECAY = 1e-3  # Increased weight decay for more regularization
    MODEL_SAVE_PATH = 'best_model.keras'
    DROPOUT_RATE = 0.5  # Increased dropout for regularization

cfg = Config()
print(f"Using TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Custom data generator class for TensorFlow
class PrescriptionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, label_path, batch_size=16, img_size=224, 
                 is_training=False, class_to_idx=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Check if label file exists
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        # Load and validate data
        try:
            # Try first with header
            self.df = pd.read_csv(label_path)
            # Check if columns exist, if not try reading without header
            if 'IMAGE' not in self.df.columns:
                self.df = pd.read_csv(label_path, header=None, 
                                    names=['IMAGE', 'MEDICINE_NAME', 'GENERIC_NAME'])
        except Exception as e:
            print(f"Error in initial CSV loading, trying alternative format: {str(e)}")
            try:
                # Alternative: skip first row and assign column names
                self.df = pd.read_csv(label_path, skiprows=1, 
                                    names=['IMAGE', 'MEDICINE_NAME', 'GENERIC_NAME'])
            except Exception as e2:
                raise ValueError(f"Failed to load CSV: {str(e2)}")
        
        # Clean up data
        self.df = self.df.dropna(subset=['IMAGE', 'MEDICINE_NAME'])
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

        # File extension handling
        self.df['IMAGE'] = self.df['IMAGE'].apply(
            lambda x: f"{x}.png" if not str(x).lower().endswith(('.png', '.jpg', '.jpeg')) else x
        )

        # Validate images
        self.valid_samples = []
        missing_files = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(data_dir, row['IMAGE'])
            if os.path.exists(img_path):
                self.valid_samples.append(row)
            else:
                missing_files.append(img_path)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} missing image files")
            if len(missing_files) < 5:  # Show a few examples
                print(f"Examples: {missing_files[:3]}")
        
        self.df = pd.DataFrame(self.valid_samples)
        if len(self.df) == 0:
            raise ValueError(f"No valid images found in {data_dir}")

        # Class handling
        if class_to_idx:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
            # Keep only samples with labels in provided classes
            self.df = self.df[self.df['MEDICINE_NAME'].isin(self.classes)]
            if len(self.df) == 0:
                raise ValueError(f"No valid samples found with provided class labels in {data_dir}")
        else:
            self.classes = sorted(self.df['MEDICINE_NAME'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        print(f"\nLoaded {len(self)} batches from {data_dir}")
        print(f"Classes: {len(self.classes)}")
        print("Class distribution:")
        class_dist = self.df['MEDICINE_NAME'].value_counts()
        print(class_dist.head())
        
        # Check for class imbalance
        if len(class_dist) > 1:
            min_samples = class_dist.min()
            max_samples = class_dist.max()
            if max_samples > 5 * min_samples:
                print(f"Warning: Significant class imbalance detected. Min: {min_samples}, Max: {max_samples}")
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_imgs = np.zeros((end_idx - start_idx, self.img_size, self.img_size, 3), dtype=np.float32)
        batch_labels = np.zeros((end_idx - start_idx), dtype=np.int32)
        
        # Fill batch
        for i, index in enumerate(batch_indices):
            row = self.df.iloc[index]
            img_path = os.path.join(self.data_dir, row['IMAGE'])
            
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Could not load {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Apply augmentations if training
                if self.is_training:
                    # Random flip
                    if np.random.random() > 0.5:
                        img = cv2.flip(img, 1)  # Horizontal flip
                    
                    if np.random.random() > 0.7:
                        img = cv2.flip(img, 0)  # Vertical flip
                        
                    # Random rotation
                    if np.random.random() > 0.5:
                        angle = np.random.randint(0, 4) * 90
                        if angle > 0:
                            (h, w) = img.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            img = cv2.warpAffine(img, M, (w, h))
                    
                    # Random brightness and contrast
                    if np.random.random() > 0.3:
                        alpha = 1.0 + 0.2 * (np.random.random() - 0.5)  # Contrast
                        beta = 0.2 * (np.random.random() - 0.5) * 127    # Brightness
                        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    
                    # Gaussian blur
                    if np.random.random() > 0.7:
                        img = cv2.GaussianBlur(img, (5, 5), 0)
                    
                    # Add noise
                    if np.random.random() > 0.8:
                        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                        img = cv2.add(img, noise)
                    
                    # Shift, scale, rotate
                    if np.random.random() > 0.5:
                        rows, cols = img.shape[:2]
                        # Random shift
                        tx = np.random.uniform(-0.1, 0.1) * cols
                        ty = np.random.uniform(-0.1, 0.1) * rows
                        # Random scale
                        scale = np.random.uniform(0.8, 1.2)
                        # Random rotation (small angles)
                        angle = np.random.uniform(-15, 15)
                        
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                        M[0, 2] += tx
                        M[1, 2] += ty
                        img = cv2.warpAffine(img, M, (cols, rows))
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                img = (img - np.array(cfg.MEAN)) / np.array(cfg.STD)
                
                batch_imgs[i] = img
                batch_labels[i] = self.class_to_idx[row['MEDICINE_NAME']]
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                # Use zeros for image and keep moving
                batch_imgs[i] = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                batch_labels[i] = self.class_to_idx[row['MEDICINE_NAME']]
        
        # One-hot encode labels
        batch_labels_one_hot = tf.keras.utils.to_categorical(
            batch_labels, num_classes=len(self.classes)
        )
        
        return batch_imgs, batch_labels_one_hot

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch for training
        self.indices = np.arange(len(self.df))
        if self.is_training:
            np.random.shuffle(self.indices)

def create_model(num_classes):
    try:
        # Load pre-trained EfficientNetV2S
        base_model = EfficientNetV2S(
            include_top=False, 
            weights='imagenet',
            input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3),
            pooling='avg'
        )
        
        # Freeze early layers to prevent overfitting
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Create model
        model = models.Sequential([
            base_model,
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Dense(640, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    except Exception as e:
        raise RuntimeError(f"Error creating model: {str(e)}")

def train_model(train_gen, val_gen, num_classes):
    # Create model
    model = create_model(num_classes)
    
    # Compile model with label smoothing
    optimizer = optimizers.Adam(
        learning_rate=cfg.LR, 
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        cfg.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=cfg.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    # Custom callback to monitor overfitting
    class OverfitMonitor(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch > 3:
                train_acc = logs.get('accuracy')
                val_acc = logs.get('val_accuracy')
                
                if train_acc - val_acc > 0.2:
                    print(f"\nWarning: Training accuracy ({train_acc:.4f}) much higher than validation accuracy ({val_acc:.4f})")
                    print("Possible overfitting detected")
    
    overfitting_cb = OverfitMonitor()
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.EPOCHS,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, overfitting_cb],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy")
    plt.legend()
    
    # Plot overfitting analysis
    plt.subplot(2, 2, 3)
    train_val_gap = [t - v for t, v in zip(history.history['accuracy'], history.history['val_accuracy'])]
    plt.plot(train_val_gap, label='Train-Val Accuracy Gap')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Mild Overfitting Threshold')
    plt.axhline(y=0.2, color='r', linestyle='-', label='Severe Overfitting Threshold')
    plt.title("Overfitting Analysis")
    plt.legend()
    
    # Plot learning rate if it's changing
    plt.subplot(2, 2, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title("Learning Rate")
        plt.yscale('log')
    else:
        # If learning rate history not available, plot val_loss again
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Validation Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

def evaluate_model(model, test_gen, class_to_idx=None):
    # Get predictions
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels (need to process entire test generator)
    true_labels = []
    for i in range(len(test_gen)):
        _, batch_labels = test_gen[i]
        true_labels.extend(np.argmax(batch_labels, axis=1))
    
    # Trim predictions to match true labels (may not be needed if batches align perfectly)
    y_pred = y_pred[:len(true_labels)]
    
    # Calculate metrics
    acc = (y_pred == true_labels).mean()
    f1 = f1_score(true_labels, y_pred, average='macro')
    
    print("\nFinal Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Class-wise metrics if classes are provided
    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_correct = {}
        class_total = {}
        
        for i in range(len(y_pred)):
            label = true_labels[i]
            class_name = idx_to_class[label]
            
            if class_name not in class_total:
                class_total[class_name] = 0
                class_correct[class_name] = 0
                
            class_total[class_name] += 1
            if y_pred[i] == label:
                class_correct[class_name] += 1
        
        print("\nClass-wise Accuracy:")
        for class_name in class_total:
            acc = class_correct[class_name] / class_total[class_name] if class_total[class_name] > 0 else 0
            print(f"{class_name}: {acc:.4f} ({class_correct[class_name]}/{class_total[class_name]})")
    
    return acc, f1

def main():
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Dataset paths - check and modify as needed
    try:
        base_path = "/kaggle/input/doctors-handwritten-prescription-bd-dataset/Doctor’s Handwritten Prescription BD dataset"
        if not os.path.exists(base_path):
            print(f"Base path not found: {base_path}")
            print("Looking for dataset in current directory...")
            # Try to find dataset in current directory or its subdirectories
            alternative_paths = [
                ".", 
                "./Doctor's Handwritten Prescription BD dataset",
                "../input/doctors-handwritten-prescription-bd-dataset/Doctor's Handwritten Prescription BD dataset"
            ]
            
            for path in alternative_paths:
                if os.path.exists(path) and any("training" in f.lower() for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))):
                    base_path = path
                    print(f"Found dataset at: {base_path}")
                    break
        
        train_dir = os.path.join(base_path, "Training/training_words")
        val_dir = os.path.join(base_path, "Validation/validation_words")
        test_dir = os.path.join(base_path, "Testing/testing_words")
        
        train_labels = os.path.join(base_path, "Training/training_labels.csv")
        val_labels = os.path.join(base_path, "Validation/validation_labels.csv")
        test_labels = os.path.join(base_path, "Testing/testing_labels.csv")
        
    except Exception as e:
        print(f"Error setting up dataset paths: {str(e)}")
        raise
    
    # Initialize data generators with try-except blocks
    try:
        print("Initializing training data generator...")
        train_gen = PrescriptionDataGenerator(
            train_dir, 
            train_labels, 
            batch_size=cfg.BATCH_SIZE, 
            img_size=cfg.IMG_SIZE, 
            is_training=True
        )
        
        print("Initializing validation data generator...")
        val_gen = PrescriptionDataGenerator(
            val_dir, 
            val_labels, 
            batch_size=cfg.BATCH_SIZE, 
            img_size=cfg.IMG_SIZE, 
            is_training=False,
            class_to_idx=train_gen.class_to_idx
        )
        
        print("Initializing test data generator...")
        test_gen = PrescriptionDataGenerator(
            test_dir, 
            test_labels, 
            batch_size=cfg.BATCH_SIZE, 
            img_size=cfg.IMG_SIZE, 
            is_training=False,
            class_to_idx=train_gen.class_to_idx
        )
    except Exception as e:
        print(f"Data generator initialization failed: {str(e)}")
        raise
    
    # Start training with error handling
    try:
        print("\nStarting model training with overfitting prevention...")
        model = train_model(train_gen, val_gen, len(train_gen.class_to_idx))
        
        # Load best model for evaluation
        print("\nLoading best model for final evaluation...")
        best_model = tf.keras.models.load_model(cfg.MODEL_SAVE_PATH)
        
        print("\nEvaluating on test set...")
        test_acc, test_f1 = evaluate_model(best_model, test_gen, train_gen.class_to_idx)
        print(f"\nFinal results - Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
        
    except Exception as e:
        print(f"Error during training or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program failed: {str(e)}")
        import traceback
        traceback.print_exc()