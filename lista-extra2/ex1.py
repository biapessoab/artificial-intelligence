import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from IPython.display import display, Image
from PIL import Image as PilImage

# Define the CNN architecture
classifier = Sequential()

# Step 1 - First Convolutional Layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 2 - Second Convolutional Layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Third Convolutional Layer
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 4 - Flatten Layer
classifier.add(Flatten())

# Step 5 - Fully Connected Layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation Configuration
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
validation_set = validation_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Define custom generator
def custom_generator(generator):
    while True:
        data = next(generator)
        yield data

repeating_training_set = custom_generator(training_set)

# Compile the CNN with a lower learning rate
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
classifier.fit(repeating_training_set, steps_per_epoch=100, epochs=5, validation_data=validation_set, validation_steps=25)

# Function to load and predict from an image
def load_and_predict_image(image_path):
    img = PilImage.open(image_path)
    img.save(image_path.replace('.bmp', '.png'))

    test_image = image.load_img(image_path.replace('.bmp', '.png'), target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    result = classifier.predict(test_image)
    
    if result[0][0] == 1:
        return 'Homer'
    else:
        return 'Bart'

# Example usage:
image_path = 'test_set/bart/bart17.bmp'
prediction = load_and_predict_image(image_path)
print(prediction)

# Evaluate the model
score = classifier.evaluate(validation_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
