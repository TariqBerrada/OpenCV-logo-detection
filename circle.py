import cv2
import numpy as np

#This scripts helps vizualize the influence of the parameters in the circle detection algorithm.
#Make sure that the frames folder contains the images: frame0.jpg, frames1.jpg...

#Frames parameters.
n_frames = 2700
frames_folder = ".\\frames"
frames_name = "frame"

def med_filter_2(image, n= 3):
    kernel = np.ones((n, n))
    #grad = cv2.morphologyEx(255-image, cv2.MORPH_GRADIENT, kernel)
    #image = 255 - cv2.erode(255-image,kernel,iterations = 1)
    opening = cv2.morphologyEx(255- image, cv2.MORPH_OPEN, kernel) 
    opening = cv2.medianBlur(opening, 7)
    #opening = cv2.bilateralFilter(opening,5,75,10)
    return 255 - opening

def detect_circles(frame, dp = .8, minDist = 50, param1 = 120, param2 = 40, maxRadius = 100):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #thresholded_img1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #thresholded_img1 = med_filter_2(thresholded_img1, 2)
    thresholded_img = 255 - cv2.Canny(frame,100,200)

    circles = cv2.HoughCircles(thresholded_img, cv2.HOUGH_GRADIENT, dp = dp, minDist = minDist, param1 = param1, param2 = param2, maxRadius = 100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(gray, (x, y), r, (0, 255, 0), 5)
            cv2.rectangle((gray), (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)
    cv2.imshow("Circle detection", gray)

def nothing(x):
    pass

# Create a window
cv2.namedWindow('params')

# Create trackbars for parametrs changing.
cv2.createTrackbar('dp', 'params', 1, 200, nothing)
cv2.createTrackbar('minDist', 'params', 1, 255, nothing)
cv2.createTrackbar('param1', 'params', 10, 255, nothing)
cv2.createTrackbar('param2', 'params', 10, 179, nothing)
cv2.createTrackbar('frame', 'params', 1, n_frames, nothing)

cv2.setTrackbarPos('param2', 'params', 70)
cv2.setTrackbarPos('frame', 'params', 255)
cv2.setTrackbarPos('dp', 'params', 5)
cv2.setTrackbarPos('minDist', 'params', 10)
cv2.setTrackbarPos('param1', 'params', 30)

hMin = sMin = vMin = hMax = vMax = 1
phMin = psMin = pvMin = phMax = pvMax = 1

while(1):
    # Get current positions of all trackbars.
    hMin = cv2.getTrackbarPos('dp', 'params')
    sMin = cv2.getTrackbarPos('minDist', 'params')
    vMin = cv2.getTrackbarPos('param1', 'params')
    hMax = cv2.getTrackbarPos('param2', 'params')
    vMax = cv2.getTrackbarPos('frame', 'params')

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (pvMax != vMax) ):
        print("dp = %f , minDist = %d, param1 = %d, param2 = %d , frame = %d" % (hMin/100 , sMin , vMin, hMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        pvMax = vMax

    # Display resulting image
    i = vMax
    frame = cv2.imread(frames_folder + "\\"+"frame" + str(i) + ".jpg")
    
    detect_circles(frame, dp = hMin/100, minDist = sMin, param1 = vMin, param2 = hMax)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
