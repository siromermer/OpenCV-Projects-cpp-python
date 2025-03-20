import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time



# Path to video  
video_path=r"../videos/plane.mp4"
video = cv2.VideoCapture(video_path)

# read only the first frame for drawing a rectangle for the desired object
ret,frame = video.read()

# STEP 1 : Chosing the target object with mouse by drawing rectangle around it.

# I am giving  big random numbers for x_min and y_min because if you initialize them as zeros whatever coordinate you go minimum will be zero 
x_min,y_min,x_max,y_max=36000,36000,0,0


def coordinat_chooser(event,x,y,flags,param):
    global go , x_min , y_min, x_max , y_max

    # when you click the right button, it will provide coordinates for variables
    if event==cv2.EVENT_LBUTTONDOWN:
        
        # if current coordinate of x lower than the x_min it will be new x_min , same rules apply for y_min 
        x_min=min(x,x_min) 
        y_min=min(y,y_min)

         # if current coordinate of x higher than the x_max it will be new x_max , same rules apply for y_max
        x_max=max(x,x_max)
        y_max=max(y,y_max)

        # draw rectangle
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),1)


    """
        if you didn't like your rectangle (maybe if you made some misscliks),  reset the coordinates with the middle button of your mouse
        if you press the middle button of your mouse coordinates will reset and you can give a new 2-point pair for your rectangle
    """
    if event==cv2.EVENT_MBUTTONDOWN:
        print("reset coordinate  data")
        x_min,y_min,x_max,y_max=36000,36000,0,0

cv2.namedWindow('coordinate_screen')
# Set mouse handler for the specified window, in this case, "coordinate_screen" window
cv2.setMouseCallback('coordinate_screen',coordinat_chooser)


while True:
    cv2.imshow("coordinate_screen",frame) # show only first frame 
    
    k = cv2.waitKey(5) & 0xFF # after drawing rectangle press ESC   
    if k == 27:
        cv2.destroyAllWindows()
        break


# STEP 2 :  Extract Features from the target object ( not from full image )

# take region of interest ( take inside of rectangle )
roi_image=frame[y_min:y_max,x_min:x_max]

# convert roi to grayscale, SIFT Algorithm works with grayscale images
roi_gray=cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY) 

# Initialize the FAST detector and BRIEF descriptor extractor
fast = cv2.FastFeatureDetector_create(threshold=20)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# detect keypoints
keypoints_1 = fast.detect(roi_gray, None)
# descriptors
keypoints_1, descriptors_1 = brief.compute(roi_gray, keypoints_1)

# draw keypoints for visualizing
keypoints_image = cv2.drawKeypoints(roi_image, keypoints_1, outImage=None, color=(0, 255, 0))
# display keypoints
plt.imshow(keypoints_image,cmap="gray")


# STEP 3 : Track Objects by using FAST Algorithm

# matcher object
bf = cv2.BFMatcher()

# Variables for FPS calculation
frame_count = 0
start_time = time.time()
     
while True :
    # reading video 
    ret,frame=video.read()

    if ret:
          # convert frame to gray scale 
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
        
        # Detect keypoints using FAST
        keypoints_2 = fast.detect(frame_gray, None)

        # Compute descriptors using BRIEF
        keypoints_2, descriptors_2 = brief.compute(frame_gray, keypoints_2)
        
        """
        Compare the keypoints/descriptors extracted from the 
        first frame(from target object) with those extracted from the current frame.
        """
        if descriptors_2 is  not None:
            matches =bf.match(descriptors_1, descriptors_2)
    
    
            for match in matches:
            
                # queryIdx gives keypoint index from target image
                query_idx = match.queryIdx
                
                # .trainIdx gives keypoint index from current frame 
                train_idx = match.trainIdx
                
                # take coordinates that matches
                pt1 = keypoints_1[query_idx].pt
                
                # current frame keypoints coordinates
                pt2 = keypoints_2[train_idx].pt
                
                # draw circle to pt2 coordinates , because pt2 gives current frame coordinates
                cv2.circle(frame,(int(pt2[0]),int(pt2[1])),5,(255,0,0),-1)
    
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
    
        cv2.imshow("coordinate_screen",frame) 
    
    
        k = cv2.waitKey(5) & 0xFF # after drawing rectangle press esc   
        if k == 27:
            cv2.destroyAllWindows()
            break
    else:
        break

video.release()
cv2.destroyAllWindows()