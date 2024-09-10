import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Use raw string literals for file paths
modelFile = r"C:/Users/Mathushan/PycharmProjects/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"C:/Users/Mathushan/PycharmProjects/face_detection/deploy.prototxt"
emotion_model_path = r"C:/Users/Mathushan/PycharmProjects/face_detection/emotion_model.hdf5"  # Updated path

try:
    # Load pre-trained face detection model
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
except Exception as e:
    print(f"Error loading face detection model: {e}")
    exit(1)

try:
    # Load pre-trained emotion detection model
    emotion_model = load_model(emotion_model_path, compile=False)
except Exception as e:
    print(f"Error loading emotion detection model: {e}")
    exit(1)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit(1)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Prepare the image for deep learning model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Perform face detection
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes around detected faces and detect emotions
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Extract the region of interest (ROI) for emotion detection
            roi = frame[startY:endY, startX:endX]
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (64, 64))  # Updated size
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                # Predict the emotion
                try:
                    emotion_prediction = emotion_model.predict(roi_gray)
                    print(f"Emotion prediction: {emotion_prediction}")  # Debug info
                    max_index = np.argmax(emotion_prediction[0])
                    emotion_label = emotion_labels[max_index]
                    print(f"Detected emotion: {emotion_label}")  # Debug info

                    # Display the label
                    label_position = (startX, startY - 10)
                    cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error in emotion prediction: {e}")

    # Display the output
    cv2.imshow("Live Face and Emotion Detection", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
