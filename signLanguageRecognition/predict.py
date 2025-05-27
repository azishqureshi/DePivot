# Azish Qureshi
# May 2025
# Signiqo AI

# As of Sunday, May 18, 2025 the model can predict every letter up to 'F' with
# exceptional accuracy. There appears to be an overfitting conflict with 'A' and 'E'
# causing the model to have minimal confusion between the two

# Due to the dataset being too large, the training and validation folders will be empty.
# A Google Drive containing the full dataset will soon be available
# The .h5 file is also not included

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Loads the trained model
model = load_model('./sign_language_model.h5')
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']

# Sets up the MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Formats the images and predicts
def classify_image(cropped_hand):
    resized_img = cv2.resize(cropped_hand, (224, 224))
    img_arr = image.img_to_array(resized_img)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)
    class_index = np.argmax(preds)
    confidence = np.max(preds)

    return CLASS_NAMES[class_index], confidence

# Starts the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flips camera (for mirror effect)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            # Gets bounding box from landmarks
            h, w, c = frame.shape
            x_list = []
            y_list = []

            for lm in handLms.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)

            xmin = int(min(x_list) * w) - 20
            xmax = int(max(x_list) * w) + 20
            ymin = int(min(y_list) * h) - 20
            ymax = int(max(y_list) * h) + 20

            # Clamps values to frame size
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > w:
                xmax = w
            if ymax > h:
                ymax = h

            # Draws the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

            # Crops and classifys images
            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size != 0:
                pred_class, conf = classify_image(hand_img)
                label = pred_class + " (" + str(round(conf * 100, 2)) + "%)"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Optionally draws landmarks
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Shows the frame
    cv2.imshow('Signiqo AI ASL Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()