import cv2
import numpy as np  # Make sure to import numpy
from PIL import Image, ImageDraw, ImageFont

# Load Haar Cascade files for face and smile detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Loop through each frame
while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(frame_grayscale, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

        # Extract the face region
        the_face = frame[y: y + h, x: x + w]

        # Convert face region to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect smiles in the face
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=45)

        # Label the face as "Smiling" if smiles are detected
        if len(smiles) > 0:
            # Use custom font with PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font_path = "Bubblegum.ttf"
            font = ImageFont.truetype(font_path, 32)
            draw.text((x, y + h + 10), "Smiling", font=font, fill=(255, 255, 255))

            # Convert PIL image back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('Smile Detector (press q to quit)', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
