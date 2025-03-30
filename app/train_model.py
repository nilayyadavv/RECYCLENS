import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


DATA_PATH = "../data/garbage_mapped"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"
TEST_DIR = "../data/test"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def create_balanced_generators():
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input
    
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.0  
    )
    
    compost_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=40,
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    recycle_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=30,
        zoom_range=0.25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Validation and test data generator (no augmentation)
    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = valid_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

train_generator, validation_generator, test_generator = create_balanced_generators()

class_index = train_generator.class_indices
class_names = list(class_index.keys())
num_classes = len(class_names)

print(f"Number of classes: {num_classes} Names: {class_names}")

with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)

disposal_mapping = {
    'compost': 'compost',
    'garbage': 'garbage',
    'recycle': 'recycle'
}

with open(os.path.join(MODEL_DIR, 'disposal_mapping.json'), 'w') as f:
    json.dump(disposal_mapping, f)

print("Created disposal mapping:", disposal_mapping)

y_integers = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced", 
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

def build_model(num_classes):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    for layer in base_model.layers:
        layer.trainable = False
    
    def squeeze_excite_block(input_tensor, ratio=16):
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = tf.keras.layers.Multiply()([input_tensor, se])
        return se
    
    x = base_model.output
    x = squeeze_excite_block(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

model, base_model = build_model(num_classes)

def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_score]
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [
    EarlyStopping(monitor='val_f1_score', patience=5, mode='max', restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model_f1.h5'), 
                   monitor='val_f1_score', mode='max', save_best_only=True),
    reduce_lr
]

print("Training initial model...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

print("Fine-tuning model...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_score]
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    initial_epoch=len(history.history['loss'])
)

def combine_history(h1, h2):
    combined = {}
    for k in h1.history.keys():
        combined[k] = h1.history[k] + h2.history[k]
    return combined

combined_history = combine_history(history, history_fine)

print("Evaluating model...")
test_results = model.evaluate(test_generator, verbose=1)
print(f"Test loss: {test_results[0]:.4f}")
print(f"Test accuracy: {test_results[1]:.4f}")
print(f"Test F1 score: {test_results[2]:.4f}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.plot(combined_history['accuracy'])
plt.plot(combined_history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--')
plt.legend(['Train', 'Validation', 'Fine-tuning start'], loc='lower right')

plt.subplot(1, 3, 2)
plt.plot(combined_history['loss'])
plt.plot(combined_history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--')
plt.legend(['Train', 'Validation', 'Fine-tuning start'], loc='upper right')

plt.subplot(1, 3, 3)
plt.plot(combined_history['f1_score'])
plt.plot(combined_history['val_f1_score'])
plt.title('Model F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.axvline(x=INITIAL_EPOCHS-1, color='r', linestyle='--')
plt.legend(['Train', 'Validation', 'Fine-tuning start'], loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
plt.close()

print("Generating predictions for confusion matrix...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

with open(os.path.join(MODEL_DIR, 'classification_report.json'), 'w') as f:
    json.dump(report, f, indent=4)

cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
plt.close()

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')
plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(MODEL_DIR, 'normalized_confusion_matrix.png'))
plt.close()

model.save(os.path.join(MODEL_DIR, 'garbage_classifier.h5'))
print(f"Model saved to {os.path.join(MODEL_DIR, 'garbage_classifier.h5')}")

def create_tflite_model():
    print("Creating TFLite model...")
    def representative_dataset_gen():
        for _ in range(10):
            # Get batch
            batch = next(iter(test_generator))[0]
            batch = batch[:min(10, len(batch))]  
            yield [batch]
    

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(MODEL_DIR, 'garbage_classifier.tflite'), 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {os.path.join(MODEL_DIR, 'garbage_classifier.tflite')}")