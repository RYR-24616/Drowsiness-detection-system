import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam    
from tensorflow.keras.utils import image_dataset_from_directory
import os 

train_dir = r"train" #insert your path here

test_dir = r"test" #insert your path here


IMG_SIZE = (96, 96) 
BATCH_SIZE = 32

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    labels = 'inferred',
    label_mode='binary',   
    shuffle=True
)
test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    labels = 'inferred',
    label_mode='binary',   
    shuffle=True
)
AUTOTUNE = tf.data.AUTOTUNE



def preprocess(image, label):
    image = tf.image.grayscale_to_rgb(image)        # Convert to 3 channels
    image = preprocess_input(image)                 # Scale to [-1, 1]
    return image, label

# Map preprocessing & performance tune
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# Build Model
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model for faster training

inputs = Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(train_ds, validation_data=test_ds, epochs=10)

save_path = r"eye_state_mobilenetv2_model.h5" #insert your path here

model.save(save_path)
print(f"Model saved to: {save_path}")
