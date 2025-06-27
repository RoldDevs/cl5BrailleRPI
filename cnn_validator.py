import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BrailleValidator:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build a CNN model for character recognition"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(26, activation='softmax')  # 26 classes for A-Z
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, train_dir, validation_dir, epochs=10, batch_size=32):
        """Train the model using data from the specified directories"""
        if not self.model:
            self.build_model()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow from directory
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(64, 64),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='sparse'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(64, 64),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='sparse'
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        return history
    
    def save_model(self, model_path='braille_cnn_model.h5'):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Build or load a model first.")
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def predict(self, image):
        """Predict the character from an image"""
        if not self.model:
            print("No model available. Build or load a model first.")
            return None
        
        # Ensure image is in the right format (64x64x1)
        if image.shape != (64, 64, 1):
            # Resize and reshape
            image = tf.image.resize(image, (64, 64))
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
            elif image.shape[-1] == 3:
                # Convert RGB to grayscale
                image = tf.image.rgb_to_grayscale(image)
        
        # Normalize
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions[0])
        
        # Convert class index to character (0=A, 1=B, etc.)
        predicted_char = chr(65 + predicted_class)  # ASCII: A=65, B=66, etc.
        confidence = predictions[0][predicted_class]
        
        return predicted_char, confidence

# Example usage
if __name__ == "__main__":
    print("This module provides CNN-based validation for braille character recognition.")
    print("It will be integrated with the main text_to_braille.py script in the future.")
    print("\nTo train the model, you can use the dataset folder structure:")
    print("dataset/")
    print("├── train/")
    print("│   ├── A/")
    print("│   ├── B/")
    print("│   └── .../")
    print("└── test/")
    print("    ├── A/")
    print("    ├── B/")
    print("    └── .../")