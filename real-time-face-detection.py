import cv2

# Creating a VideoCapture object to initialize webcam
cap = cv2.VideoCapture(0)

# Creating a Cascade classifier object
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Reading video frames
while True:

    ret,frame = cap.read()

    # Reading the image as grayscale image
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Search the co-ordinates of the image
    detections = cascade_classifier.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)

    # Drawing recangle
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)

    # Opens a window to display the image
    cv2.imshow('frame', frame)

    # Closes window on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()

# Closes the window based on waifortkey parameter
cv2.destroyAllWindows()
