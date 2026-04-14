import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = []
labels = []
label_dict = {}

dataset_path = "dataset"

for i, person in enumerate(os.listdir(dataset_path)):
    label_dict[i] = person
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name))
        img = cv2.resize(img, (64,64))
        data.append(img)
        labels.append(i)

data = np.array(data) / 255.0
labels = to_categorical(labels)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, labels, epochs=10)

model.save("model.keras")

print("Training Completed ✅")
print(label_dict)