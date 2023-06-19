import cv2
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# load the pre-trained model and labels
model = load_model('model.h5')
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# load the face detection classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/emotion_detection', methods=['POST'])
def detect_emotion():
    # read the image file from the request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the detected faces
    for (x, y, w, h) in faces:
        # extract the face ROI, resize it, and preprocess it
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # make a prediction on the ROI, then lookup the class label
        preds = model.predict(roi_gray)[0]
        label = labels[preds.argmax()]

    # return the detected emotion
    return jsonify({'emotion': label}), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run('0.0.0.0', port=5050)
