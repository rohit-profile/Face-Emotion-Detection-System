from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import base64   
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load face detector & emotion model
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
model = load_model('fer_cnn_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion_text = ''
    image_data = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                emotion_text = "Invalid image uploaded"
                return render_template('index.html', emotion_text=emotion_text)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                emotion_text = 'No faces detected'
            else:
                emotions_detected = []

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi_gray, (32, 32))
                    roi_normalized = roi_resized.astype('float32') / 255.0
                    roi_normalized = np.expand_dims(roi_normalized, axis=-1)
                    roi_input = np.expand_dims(roi_normalized, axis=0)

                    preds = model.predict(roi_input, verbose=0)[0]
                    label = emotion_labels[np.argmax(preds)]
                    emotions_detected.append(label)

                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                emotion_text = 'Detected emotions: ' + ', '.join(emotions_detected)

                _, buffer = cv2.imencode('.png', img)
                image_bytes = buffer.tobytes()

               
                image_data = base64.b64encode(image_bytes).decode('utf-8')
                image_data = f"data:image/png;base64,{image_data}"

    return render_template('index.html', emotion_text=emotion_text, image_data=image_data)


if __name__ == "__main__":
    app.run(debug=True)
