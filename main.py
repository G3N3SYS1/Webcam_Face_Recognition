# main.py
import json
import cv2
import os
from face_recog import Face_recognition

selected_face_index = None

# Encode faces from a folder
sfr = Face_recognition()

current_dir = os.path.dirname(__file__)  # Get the directory of the current file
ANC_PATH = os.path.join(current_dir, "images")

# Loads encoded faces from images folder
sfr.load_encoding_images(ANC_PATH)

# Will add feature to improve load times by writing faces metadata into json file
# and reads from json file

# Select device webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Update number of images on each iteration (optional)
    num_images = len(os.listdir(ANC_PATH))

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    unknown_faces =[]
    unknown_locations = []
    for i, (face_loc, name) in enumerate(zip(face_locations, face_names)):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Extract the face region
        face_image = frame[y1:y2, x1:x2]

        # Draw rectangle and name for all faces
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        if name == "Unknown":
            unknown_faces.append(face_image)
            unknown_locations.append(face_loc)

        # Handle saving unknown faces (if any)
        if len(unknown_faces) > 0:
            if len(unknown_faces) == 1:  # Only 1 unknown face
                # Use the first unknown face for saving
                unknown_image = unknown_faces[0]
                unknown_loc = unknown_locations[0]
                y1, x2, y2, x1 = unknown_loc[0], unknown_loc[1], unknown_loc[2], unknown_loc[3]

                if cv2.waitKey(1) & 0xFF == ord('r'):
                    # Save filename according to user's input
                    filename = input("Enter name of individual: ")
                    imgname = os.path.join(ANC_PATH, f"{filename}.jpg")
                    # Crops and save Face within the rectangle
                    cv2.imwrite(imgname, unknown_image)
                    # Optionally, update encodings after saving a new image
                    sfr.load_encoding_images(ANC_PATH)
                    print(f"Saved image: {imgname}")
            else:
                print(f"Multiple unknown faces detected. Please try again!")
                #Maybe can implement face selection, which one to save.
                break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()