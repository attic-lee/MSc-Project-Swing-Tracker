#########################################
#To run this programme, (1) install the requirements, (2) Open cmd or terminal run "python run.py" OR open cmd and run "streamlit run Home.py"
########################################
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import streamlit as st
import tempfile
from PIL import Image


#video path 
#video_demo = r"C:\Users\Attic\OneDrive - University of Greenwich/MSc Data Science/MSc Project/Code/examples/pro_ex.mp4"
video_demo = r"examples/pro_ex.mp4"

st.title("Tennis swing tracker🎾")

st.markdown('In this application we are using **MediaPipe** for creating a Pose Estimation. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
st.video(video_demo)

st.markdown('''
          # About Me \n 
            I am Attic Lee, student of MSc Data Science, University of Greenwich.\n
            \n
            I am a tennis lover and Artificial Intelligence Enthusiast so that I combined my interest with my final project.           
            ''')


st.markdown('''
          # About this project \n 
            This Project aimed to provide a tool for tennis players and coaches to analyse their storke such as serve using computer vision and pose estimation. \n

                  
            ''')

st.sidebar.success("Start analysing your game with 'Swing Tracker' !")