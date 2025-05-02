import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

#load the trained model
model = load_model('facemask_model.h5')

#set image size used during training
Img_size=150

#Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#Start video capture
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while  cap.isOpened():
        ret, frame = cap.read()
        if not ret :
            break

        #convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results= face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                face = frame [y: y+h_box, x:x+w_box]

                try:
                    #preprocess face
                    face_resized = cv2.resize(face, (150,150))
                    face_array= np.expand_dims(face_resized/255.0, axis=0)

                    #predict the mask status
                    prediction = model.predict(face_array)
                    # Print the prediction score to the terminal or console
                    print("Prediction score:", prediction)
                    label = "With Mask" if prediction >= 0.5 else "Without Mask"
                    color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

                    #draw rectangle and label
                    cv2.rectangle(frame, (x,y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(frame, label, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except:
                    pass

        cv2.imshow('Mask Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()