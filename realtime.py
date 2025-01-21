from tensorflow import keras
import cv2
import numpy as np
import os, json
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# TensorFlow Components
Load_Model = keras.models.load_model
L2 = keras.regularizers.l2
Model = keras.models.Model
Dense = keras.layers.Dense
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
to_categorical = keras.utils.to_categorical

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

IMG_SIZE = 128
ANNOTATIONS = config["annotations_dir"]
IMAGES = config["images_dir"]
MODEL_FILE = config["model_file"]
LOG_FILE = config["log_file"]
label_dict = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}

# Parse Annotations
def parse_annotations(annotations_dir, images_dir):
    data = []
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
        try:
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
        except ET.ParseError:
            print(f"Error parsing {xml_file}, skipping...")
            continue
        root = tree.getroot()
        filename = root.find('filename').text
        filepath = os.path.join(images_dir, filename)
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append({'class': name, 'bbox': (xmin, ymin, xmax, ymax)})
        data.append({'filepath': filepath, 'objects': objects})
    return data

dataset = parse_annotations(ANNOTATIONS, IMAGES)

# Load and Preprocess Data
def load_data(dataset, img_size):
    images, labels = [], []
    for data in dataset:
        filepath = data['filepath']
        for obj in data['objects']:
            img = cv2.imread(filepath)
            if img is None:
                print(f"Error reading {filepath}, skipping...")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            xmin, ymin, xmax, ymax = obj['bbox']
            cropped_img = img[ymin:ymax, xmin:xmax]
            resized_img = cv2.resize(cropped_img, (img_size, img_size))
            normalized_img = resized_img / 255.0
            images.append(normalized_img)
            labels.append(label_dict[obj['class']])
    return np.array(images), np.array(labels)

# Evaluation Metric Function
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy

X, Y = load_data(dataset, IMG_SIZE)
Y = to_categorical(Y, num_classes=3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define Model
def build_model(input_shape, num_classes):
    base_model = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu', kernel_regularizer=L2(1e-4))(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=L2(1e-4))(x)
    return Model(inputs=base_model.input, outputs=output)

input_shape = (IMG_SIZE, IMG_SIZE, 3)
model = build_model(input_shape, 3)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and Save Model
model.fit(X_train, Y_train, epochs=10, validation_split=0.2)
model.save('face_mask_model.keras')

# Real-Time Detection
def preprocess_frame(frame, bbox):
    xmin, ymin, xmax, ymax = bbox
    cropped_img = frame[ymin:ymax, xmin:xmax]
    resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=0)

model = Load_Model(MODEL_FILE)

def real_time_detection():
    # Haar Cascade for Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Webcam Initialization
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    label_dict_reverse = {v: k for k, v in label_dict.items()}

    # For evaluation
    all_true_labels = []
    all_predicted_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            bbox = (x, y, x + w, y + h)
            preprocessed_img = preprocess_frame(frame, bbox)
            predictions = model.predict(preprocessed_img)
            label_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            label = label_dict_reverse[label_idx]

            # Simulate true label (in real scenario, this should come from a labeled dataset)
            true_label_idx = label_idx  ### Replace with actual label in production
            all_true_labels.append(true_label_idx)
            all_predicted_labels.append(label_idx)

            # Draw Bounding Box and Label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display Real-Time Feed
        cv2.imshow('Real-Time Face Mask Detection with Metrics', frame)

        # Calculate and Display Metrics
        if len(all_true_labels) > 0:
            precision, recall, f1, accuracy = calculate_metrics(all_true_labels, all_predicted_labels)
            print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, Accuracy: {accuracy:.2f}")
            # Save metrics to a file
            with open(LOG_FILE, 'a') as f:
                f.write(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, Accuracy: {accuracy:.2f}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run Real-Time Detection
real_time_detection()
