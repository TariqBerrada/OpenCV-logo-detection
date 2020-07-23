import cv2
import numpy as np
import os

def clear_folder(folder_path):
    filelist = [ f for f in os.listdir(folder_path)]
    for f in filelist:
        os.remove(os.path.join(folder_path, f))

def save_frames(video_path, output_folder=".\\frames", frame_name = "frame"):
    """
    Saves video frames to output folder.
    """

    cap = cv2.VideoCapture(video_path)
    ref_path = ".\\"
    cond = True
    i = 0
    print("Saving frames to ouput folder.")
    while cond:
        cond, frame = cap.read()
        path = os.path.join(ref_path, output_folder, frame_name +str(i)+".jpg")
        cv2.imwrite(path, frame)
        i+=1
    print("Done extracting frames.")
    return i-2

def med_filter_2(image, n= 3):
    """
    Refinement filter used to detect circles, not used below because canny gave better results.
    """
    
    kernel = np.ones((n, n))
    #grad = cv2.morphologyEx(255-image, cv2.MORPH_GRADIENT, kernel)
    #image = 255 - cv2.erode(255-image,kernel,iterations = 1)
    opening = cv2.morphologyEx(255- image, cv2.MORPH_OPEN, kernel) 
    opening = cv2.medianBlur(opening, 7)
    #opening = cv2.bilateralFilter(opening,5,75,10)
    return 255 - opening

def detect_circles(frame, dp = .8, minDist = 50, param1 = 25, param2 = 45, maxRadius = 100):
    """
    Detect circles in frame returns the circle coordinates x_o, y_o, r.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #thresholded_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #thresholded_img = med_filter_2(thresholded_img, 2)
    thresholded_img = 255 - cv2.Canny(frame,100,200)

    circles = cv2.HoughCircles(thresholded_img, cv2.HOUGH_GRADIENT, dp = dp, minDist = minDist, param1 = param1 , param2 = param2 ,minRadius = 2, maxRadius = maxRadius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    
    return circles

def detect_color(frame):
    """
    Generates color mask for the red used in the logo.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    low_red = np.array([0,0,0])
    high_red = np.array([179,105,255])
    #low_red = np.array([0,100,100])
    #high_red = np.array([13,255,255])

    mask = cv2.inRange(hsv_frame, low_red, high_red)

    return mask

def overlaps(subsection, mask, threshold = 100):
    """
    Returns a probability of the subsection holding a shape similar to that of the logo.
    """

    new_subs = cv2.resize(subsection, (32, 32))
    new_subs = cv2.cvtColor(new_subs, cv2.COLOR_BGR2GRAY)
    _, new_subs = cv2.threshold(new_subs,127,255,cv2.THRESH_BINARY)

    new_mask = cv2.resize(mask, (32, 32))
    new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
    _, new_mask = cv2.threshold(new_mask,127,255,cv2.THRESH_BINARY)

    s = 1 - np.sum((np.abs(new_subs-new_mask)-127.5)/127.5)/(32*32)

    #Adding a second condition
    x = np.sum(255-cv2.Canny(subsection, 100, 200))/(255*np.shape(subsection)[1]*np.shape(subsection)[0])

    return s

def get_boxes(frame, idx, threshold=200, n=3):
    """
    Performs circle detection, color detection and overlap tests in order to generate boxes corresponding to detections.
    Returns frame list of bounding boxe coordinates for each detection.
    """

    circles = detect_circles(frame)
    color_mask = detect_color(frame)
    detection_masks = [cv2.imread(".\\detection_masks\\"+str(i)+".jpg") for i in range(1, 5)]
    ly, lx, _ = np.shape(frame)

    boxes_list = []
    coordinates = None
    if circles is not None:
        for circle in circles:
            x, y, r = circle
            
            xmin, xmax, ymin, ymax = max(x-n-1, 0), min(x+n,lx), max(0, y-n-1), min(y+n, ly)

            neighborhood = color_mask[ymin:ymax, xmin:xmax]

            score = np.mean(neighborhood)
            if score < threshold:
                #x, y, w, h
                coordinates = [idx, int(x-1.2*r), int(y-1.2*r), int(2.4*r), int(2.4*r)]
                if min(coordinates[1:]) > 0 and coordinates[2] + coordinates[4] < lx and coordinates[1] + coordinates[3] < ly:
                    s = 0
                    for i in range(4):
                        s = max(s, overlaps(frame[coordinates[2]:coordinates[2] + coordinates[4], coordinates[1]:coordinates[1]+coordinates[3]], detection_masks[i]))
                    #if s > 1.6:
                        boxes_list.append(coordinates)

    return boxes_list   

def convert_coords(coordinates_i):
    """
    Converts coordinates list to string.
    """
    return ", ".join([str(c) for c in coordinates_i]) + "\n"

def write_coords(coordinates, filename):
    """
    Writes list of string coordinates to filename.txt
    """
    print("trying to write coordinates")
    output = open(filename+".txt", "w")
    output.writelines(coordinates)
    output.close()
    print("Done writing to %s"%(filename + ".txt"))

def detect(entry_vid, output_filename="out"):
    """
    Performs all of the operations necessary to get the bounding boxes corresponding to each frame on a text file as requested.
    """

    clear_folder(".\\frames\\")
    n_frames = save_frames(entry_vid)

    print("Processing frames and writing to file.")
    coordinates = ["frame_id | x | y | w | h \n"]
    for i in range(n_frames):
        if i%50 == 0:
            print("Frame %d/%d."%(i, n_frames))

        frame = cv2.imread(".\\frames\\frame" + str(i) + ".jpg")
        c_i = get_boxes(frame, i)
        for c in c_i:
            x = convert_coords(c)
            coordinates.append(x)

    write_coords(coordinates, ".\\output\\" + output_filename)
    print("Done writing coordinates.")
    print("Deleting generated frames.")
    clear_folder(".\\frames\\")
    print("Done")

