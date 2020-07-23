import cv2
import numpy as np

#This script shows the color detection filter for differnt values of hsv bounds.
#Make sure that the frames folder contains the images: frame0.jpg, frames1.jpg...

n_frames = 2700

def nothing(x):
    pass

# Create a window
cv2.namedWindow('filter')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'filter', 0, 179, nothing)
cv2.createTrackbar('SMin', 'filter', 0, 255, nothing)
cv2.createTrackbar('VMin', 'filter', 0, 255, nothing)
cv2.createTrackbar('HMax', 'filter', 0, 179, nothing)
cv2.createTrackbar('SMax', 'filter', 0, 255, nothing)
cv2.createTrackbar('VMax', 'filter', 0, 255, nothing)
cv2.createTrackbar('frame', 'filter', 0, n_frames, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'filter', 179)
cv2.setTrackbarPos('SMax', 'filter', 255)
cv2.setTrackbarPos('VMax', 'filter', 255)
cv2.setTrackbarPos('frame', 'filter', 1)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0
frm = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'filter')
    sMin = cv2.getTrackbarPos('SMin', 'filter')
    vMin = cv2.getTrackbarPos('VMin', 'filter')
    hMax = cv2.getTrackbarPos('HMax', 'filter')
    sMax = cv2.getTrackbarPos('SMax', 'filter')
    vMax = cv2.getTrackbarPos('VMax', 'filter')
    frm = cv2.getTrackbarPos('frame', 'filter')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Load image
    image = cv2.imread(".\\frames\\frame" + str(frm) + ".jpg")
    x = 255 - cv2.Canny(image, 100, 200)
    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(x, x, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('filter', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()