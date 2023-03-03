from tkinter import W
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import streamlit as st
import tempfile
from PIL import Image
import os 
import ffmpeg


## Pose Module
st.sidebar.subheader('Parameters')
detection_confidence = st.sidebar.slider(
'Min Detection Confidence', 
min_value =0.0,
max_value = 1.0,
value = 0.5)

tracking_confidence = st.sidebar.slider(
'Min Tracking Confidence', 
min_value = 0.0,
max_value = 1.0,
value = 0.5)

alpha = st.sidebar.slider(
"Video Opacity",
min_value = 0.0,
max_value = 1.0,
value = 1.0)

beta = st.sidebar. slider(
"Motion Path Opacity",
min_value = 0.0,
max_value = 1.0,
value = 1.0)

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=detection_confidence, trackCon=tracking_confidence):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
       #self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
       #                               self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(255,255,10), thickness=3, circle_radius=3),
                                           self.mpDraw.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=1))
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


##################################
##################################
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

###############################
###############################
@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

###############################
###############################

st.title("Tennis swing tracker🎾")

st.markdown(
    """
    This is an computer vision app that compare your tennis serve swing with the professionals like Roger Federer and Rafael Nadal.

    
    
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;   
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.markdown('---')

#parameters
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
detector = poseDetector()

drawColor = (0, 255, 0) #
drawColorInput = (0, 255, 255)
brushThickness= 10
xp, yp = 0, 0 #coordinate
imgCanvas = np.zeros((1080, 1920, 3), np.uint8) #Create canvas for drawing

drawing_spec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)

#video path 
#video_demo = r"C:\Users\Attic\OneDrive - University of Greenwich/MSc Data Science/MSc Project/Code/examples/pro_ex.mp4"
video_demo = r"examples/pro_ex.mp4"
#path = r"C:\Users\Attic\OneDrive - University of Greenwich\MSc Data Science\MSc Project\Code\Input\Ivo_Karlovic_speeded.mp4"
path = r"Input\Ivo_Karlovic_speeded.mp4"

cap = cv2.VideoCapture(path)



#Demo video
with st.container():
    st.markdown(' ## Examples')
    st.video(video_demo)

st.markdown(' ## Model')

stframe = st.empty()
video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
tfflie = tempfile.NamedTemporaryFile(delete=False)


if not video_file_buffer:
        cap = cv2.VideoCapture(path)
        tfflie.name = path
else:
    tfflie.write(video_file_buffer.read())
    cap = cv2.VideoCapture(tfflie.name)


st.sidebar.text('Input Video')
st.sidebar.video(tfflie.name)

#get width, height, fps from capture
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

fps = 0
i = 0

# Download the frame
fourcc = cv2.VideoWriter_fourcc(*'avc1')
w = int(cap.get(3))
h = int(cap.get(4))
output_img = 'output/output.mp4'
output_img = cv2.VideoWriter(output_img, fourcc, 20.0, (w,h))

# Download the canvas
output_canv = 'output_paths/output_path.mp4'
output_canv = cv2.VideoWriter(output_canv, fourcc, 20.0, (w,h))

kpi1, kpi2 = st.columns(2)
tab1, tab2, tab3, = st.tabs(["Examples", "Result", "Motion Path Comparison"])

with kpi1:
    st.markdown("**FrameRate**")
    kpi1_text = st.markdown("0")

with kpi2:
    st.markdown("**Image Width**")
    kpi3_text = st.markdown("0")

st.markdown("<hr/>", unsafe_allow_html=True)

with tab1:
    st.text('Examples of few professional players')
    st.video(video_demo)

with tab2:  
    st.header('Result')
    with mpPose.Pose() as pose:
        prevTime = 0
        while cap.isOpened():
            i +=1
            ret, img = cap.read()
            if ret == True:

                img = detector.findPose(img)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(imgRGB)
                img.flags.writeable = True
                lmList = detector.findPosition(img, draw=False)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    shoulder = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    angle = round(calculate_angle(shoulder, elbow, wrist),1)

                    # Visualize angle
                #  cv2.putText(img, str(angle), 
                #              tuple(np.multiply(elbow, [1800, 800]).astype(int)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                #                    )
                except:
                    pass   

                if len(lmList) !=0:
                    #wrist position
                    x1, y1 = lmList[16][1:]
                    # Add a circle in the wrist position
                    cv2.circle(img, (x1, y1), 5, drawColorInput, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    
                    #Draw a line that tracks the location of the wrist
                    cv2.line(img, (xp, yp), (x1, y1), drawColorInput, 10)
                    if drawColor == (0, 0, 0) :
                        cv2.line(img, (xp, yp), (x1, y1), drawColorInput, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColorInput, brushThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColorInput, 10)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColorInput, 10)

                    xp, yp = x1, y1

                #Blend canvas with the frame
                    img = cv2.addWeighted(img,alpha,imgCanvas,beta,0) 
                #show the video
                    #imgshow = cv2.imshow("Image", img)
                    #canvas_show = cv2.imshow("Canvas", imgCanvas)
                # Save the video
                    output_img.write(np.uint8(img))
                    output_canv.write(np.uint8(imgCanvas))
            else:
                cap.release()
                output_img.release()
                output_canv.release()
                cv2.destroyAllWindows()
                break   
            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime 

            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            img = cv2.resize(img,(0,0),fx = 0.8 , fy = 0.8)
            img = image_resize(img, width = 640)
            #canvas = image_resize(imgCanvas, width = 640)
            stframe.image(img,channels = 'BGR',use_column_width=True)


    st.text('Video Processed')
    output_video = open('output/output.mp4','rb')
    out_bytes = output_video.read()
    video = st.video(out_bytes) 

    st.text('Motion Path')
    canvas_vid = open('output_paths/output_path.mp4','rb')
    canvas_bytes = canvas_vid.read()
    vid = st.video(canvas_bytes)


with tab3:
    stframe2 = st.empty()
    #pro example motion
    #path2 = r"C:\Users\Attic\OneDrive - University of Greenwich\MSc Data Science\MSc Project\Code\output_path.mp4"
    path2 = r"output_paths/output_path.mp4"
    video2 = cv2.VideoCapture(path2)

    #User input motion
    #path3 = r"C:\Users\Attic\OneDrive - University of Greenwich\MSc Data Science\MSc Project\Code\moition path\example_karlovic_path.mp4"
    path3 = r'output_paths/example_karlovic_path.mp4'
    video3 = cv2.VideoCapture(path3)

    new_width = int(video2.get(3))
    new_height = int(video2.get(4))

    vid3_width = int(video3.get(3))
    vid3_height =int(video3.get(4))

    #Comparison video
    output2 = r'output_paths/result_combined_path.mp4'
    output2 = cv2.VideoWriter(output2, fourcc, 20.0, (new_width,new_height))

    black = np.zeros((new_height,new_width), dtype = "uint8")

    while True:
        ret1, frame1 = video2.read()
        ret2, frame2 = video3.read()

        if ret1 == False or ret2 == False:
            video2.release()
            video3.release()
            output2.release()
            break
        
        resultFrame = cv2.addWeighted(frame1,1,frame2,1,0)
        output2.write(np.uint8(resultFrame))
        cv2.imshow('result', resultFrame)



    st.text('Comparison of exmaple'' motion and user\'s motion')
    result_vid = open('output_paths/result_combined_path.mp4','rb')
    result_bytes = result_vid.read()
    comparison_path = st.video(result_bytes)


## Remove footer 
# hide_menu_style = """
#         <style>
#         #mainmenu {visibility :hidden; }
#         footer {visibility :hidden; }
#         </style>
#         """
#st.markdown(hide_menu_style, unsafe_allow_html=True)