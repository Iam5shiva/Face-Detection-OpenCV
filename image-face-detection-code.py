import cv2

# Creating a Cascade classifier object
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Reading the image as it is
img = cv2.imread("img2.jpg")

# Reading the image as grayscale image
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Search the co-ordinates of the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)

# Drawing recangle on image
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),15)

# reshaping the image to display properly on the screen
resized = cv2.resize(img,(int(img.shape[1]/7), int(img.shape[1]/7)))

# Opens a window to display the image
cv2.imshow("Win", resized)

# Wait until user presses a key
cv2.waitKey(0)

# Closes the window based on waifortkey parameter
cv2.destroyAllWindows()
