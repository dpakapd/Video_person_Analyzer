import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall


# Hyperparameters
batch_size = 32
epochs = 10
learning_rate = 1e-4
image_size = (224, 224)
l2_reg = 0.01
unfreeze_from_layer = 140

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
)

# Model architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze and fine-tune some layers of the base model
for layer in base_model.layers[:unfreeze_from_layer]:
    layer.trainable = False

for layer in base_model.layers[unfreeze_from_layer:]:
    layer.trainable = True

# Compile the model
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping],
)

# Save the model
model.save('resnet50_classifier_updated.h5')

# Evaluate the model on test data
#test_loss, test_acc = model.evaluate(test_generator)
#print(f"Test accuracy: {test_acc:.2f}")

# Evaluate the model on test data
test_results = model.evaluate(test_generator)
print("Test results:")
for i in range(len(model.metrics_names)):
    print(f"{model.metrics_names[i]}: {test_results[i]:.2f}")

