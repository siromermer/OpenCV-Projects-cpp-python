import cv2
import numpy as np



# path to video  
video_path=r"videos/helicopter3.mp4"  

video = cv2.VideoCapture("videos/helicopter3.mp4")

# read only the first frame to draw a rectangle for object
ret,frame = video.read()

# I am giving  big random numbers for x_min and y_min because if you initialize them as zeros whatever coordinate you go minimum will be zero 
x_min,y_min,x_max,y_max=36000,36000,0,0

# function for choosing min and max coordinates 
def coordinat_chooser(event,x,y,flags,param):
    global go , x_min , y_min, x_max , y_max

    # when you click the right button it is gonna give variables some coordinates
    if event==cv2.EVENT_RBUTTONDOWN:
        
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
        if you press the middle button of your mouse coordinates will reset and you can give new 2-point pair for your rectangle
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
        break


# STEP 2: The program detects the color of the object that the user selected.

# inside of rectangle that user draw 
object_image=frame[y_min:y_max,x_min:x_max,:]

hsv_object=cv2.cvtColor(object_image,cv2.COLOR_BGR2HSV)    

# cx and cy are the centers of the rectangle that the user chose 
height, width, _ = hsv_object.shape
cx = int(width / 2)
cy = int(height / 2)

# take the center pixel to find out which rectangle color.
pixel_center = hsv_object[cy, cx]
hue_value = pixel_center[0] # axis 0 is hue values


# from hue_value find color
color =str()
if hue_value < 5:
    color = "red"
elif hue_value < 22:
    color = "orange"
elif hue_value < 33:
    color = "yellow"
elif hue_value < 78:
    color = "green"
elif hue_value < 131:
    color = "blue"
elif hue_value < 170:
    color = "violet"
else:
    color = "red"

# hue dict 
hue_dict={ "red":[[[0, 100, 100]],[10, 255, 255]],
           "orange":[[10, 100, 100],[20, 255, 255]],
           "yellow":[[20, 100, 100],[30, 255, 255]],
           "green":[[50, 100, 100],[70, 255, 255]],
           "blue":[[110,50,50],[130,255,255]],
           "violet":[[140, 50, 50],[170, 255, 255]]}

# find the upper and lower bounds of the image's color
lower_bound , upper_bound = np.asarray(hue_dict[color][0]) , np.asarray(hue_dict[color][1]) # lower and upper bound sequentially

print(f"detected color : {color}" )


# STEP 3: Tracking object.

# This time display all video, not just the first frame 
# (in the first part only the first frame was displayed on the screen because the user was choosing an object by drawing a rectangle)   
video=cv2.VideoCapture(video_path)

# we need first frame for creating roi(region of interest)
ret,cap = video.read()

# coordinates that the user gives with his mouse 
x=x_min
y=y_min
w=x_max-x_min
h=y_max-y_min

track_window = (x, y, w, h)

# set up the ROI for tracking
roi = cap[y:y+h, x:x+w]

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# use lower_bound and upper_bound  inside of inRange function
mask = cv2.inRange(hsv_roi, lower_bound,upper_bound )
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iterations or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:

    ret, frame = video.read()

    cv2.putText(frame,f"detected color : {color}" , (25,25),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),1)


    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply mean-shift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        
        # Draw it on the image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
 
        cv2.imshow('img2',img2)

        if cv2.waitKey(30) & 0xff == 27:
            break


cv2.destroyAllWindows()