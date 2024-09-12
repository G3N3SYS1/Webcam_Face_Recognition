# main.py
import json
import cv2
import os
from face_recog import Face_recognition

# Encode faces from a folder
sfr = Face_recognition()

current_dir = os.path.dirname(__file__)  # Get the directory of the current file
ANC_PATH = os.path.join(current_dir, "images")

sfr.load_encoding_images(ANC_PATH)

# Select device webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Update number of images on each iteration (optional)
    num_images = len(os.listdir(ANC_PATH))

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Extract the face region
        face_image = frame[y1:y2, x1:x2]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('r'):

        # Save filename according to user's input
        filename = input("Enter name of individual: ")
        imgname = os.path.join(ANC_PATH, f"{filename}.jpg")
        # Crops and save Face within the rectangle
        cv2.imwrite(imgname,face_image)

        # Optionally, update encodings after saving a new image
        sfr.load_encoding_images(ANC_PATH)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()