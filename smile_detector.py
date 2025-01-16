import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    exit()

# Loop through each frame
while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

        # Extract the face region
        the_face = frame[y: y + h, x: x + w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=45)

        if len(smiles) > 0:
            # Use custom font with PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font_path = "Bubblegum.ttf"
            font = ImageFont.truetype(font_path, 32)
            draw.text((x, y + h + 10), "Smiling", font=font, fill=(255, 255, 255))

            # Convert PIL image back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('Smile Detector (press q to quit)', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
