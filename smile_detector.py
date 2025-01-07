import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab Webcam feed
webcam = cv2.VideoCapture(0)

# Show the current frame
while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
         break
    
    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    # Run face detection within each of those faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)
        
        # Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y: y + h, x : x + w]
        
        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        
        # Detect smiles
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
        
        # Label this face as smilling
        if len(smiles) > 0:
            cv2.putText(
                frame, 
                'Smilling', 
                (x, y + h + 40), 
                fontScale=1.3, 
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                color=(255,255,255)
            )

    # Show current frame
    cv2.imshow('Smile Detector', frame )

    # Go to next frame in 1 ms
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up: tell the OS that the app is done using the webcam
webcam.release()

# Close all windows
cv2.destroyAllWindows()