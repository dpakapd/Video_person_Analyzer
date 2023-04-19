import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imageio

# Load the trained model
model = load_model('resnet50_classifier_updated.h5')

gif_data = [] # Empty list to collect GIF frames

# Load a pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to preprocess and predict the age group
def predict_age_group(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = img_to_array(face_img)
    face_img = preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    prediction = model.predict(face_img)
    return np.argmax(prediction)

# Process the video
video_capture = cv2.VideoCapture('test-video/test_child_2.mp4')
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        age_group = predict_age_group(face_img)
        label = 'Child' if age_group == 1 else 'Adult'

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gif_data.append(frame) #Add the frame to the GIF_DATA in order to create a GIF
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Saving to GIF")
        imageio.mimsave("classifier.gif", gif_data, fps = 60)
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
