# Drowsiness-Detector
Drowsiness Detector is a computer vision system that automatically detects if the user drowsiness in real-time from a live video stream and then alert the user with an alarm notification. 

This repository is based on the tutorial by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Install and set up Python 3.
1. Install [cmake](https://github.com/Kitware/CMake/releases/download/v3.13.3/cmake-3.13.3-win64-x64.zip) in your system

## Running the application

1. Clone the repository. 

    ```
    git clone https://github.com/bedangSen/Drowsiness-Detector.git
    ```
    
1. Move into the project directory. 

    ```
    cd Drowsiness-Detector
    ```
 
1. (Optional) Running it in a virtual environment. 

   1. Downloading and installing _virtualenv_. 
   ```
   pip install virtualenv
   ```
   
   2. Create the virtual environment in Python 3.
   
   ```
    virtualenv -p C:\Python37\python.exe test_env
   ```    
   
   3. Activate the test environment.     
   
        1. For Windows:
        ```
        test_env\Scripts\Activate
        ```        
        
        2. For Unix:
        ```
        source test_env/bin/activate
        ```    

1. Install all the required libraries, by installing the requirements.txt file.

    ```
    pip install -r requirements.txt
    ```
    
1. Installing the dlib library.
     
    1. If you are using a Unix machine, and are facing some issues while trying to install the dlib library, follow [this guide](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).  
    
    1. If you are using a Windows machine, install cmake and restart your terminal. 
    
1. Run the application.

    ```
    python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
    ```
    
## Building the Code 

1. We begin by creating a python file and naming it something like `drowsiness_detector.py`. Then we start off the application by importing the required libraries. 
    
    ```
    # import the necessary packages
    from scipy.spatial import distance as dist
    from imutils.video import VideoStream
    from imutils import face_utils
    from threading import Thread
    import numpy as np
    import playsound
    import argparse
    import imutils
    import time
    import dlib
    import cv2
    ```
    
 1. If you do not have any of the lbraries, like the imutils, dlib or the playsound library, you can install them using the following command:
 
    ```
    pip install imutils
    ```
    
    ```
    pip install playsound
    ```
    
    ```
    pip install dlib
    ```
    
 1. Next we write the function for sounding the alarm. The function takes in the path to the audio file on the local system, and plays it.
 
    ```
    def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)
    ```
    
 1. We then define the function for calculating the eye aspect ratio. This function calculates the distance between the vertical eye landmarks and the horizontal eye landmarks.
 
    ```
    def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
    ```
    
  1. Next we will write the code for parsing the arguments the user will provide while running the application. 
  
    ```
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type=str, default="",
        help="path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0,
        help="index of webcam on system")
    args = vars(ap.parse_args())
    ```
    
  1. Next we define a few consants for the application:
    
    1. `EYE_AR_THRESH` - This sets the threshold for what counts as closed eyes. If the eye aspect ratio falls below this value, a counter initiates for the number of frames the eye has been closed. 
    1. `EYE_AR_CONSEC_FRAMES` - This sets the number of frames before the alarm is sounded. For a more sensitive application, set this value to a lower amount. For a less sensitive application, set the value to higher amount.
    
## Built With

* [OpenCV Library](https://opencv.org/) - Most used computer vision library. Highly efficient. Facilitates real-time image processing.
* [imutils library](https://github.com/jrosebr1/imutils) -  A collection of helper functions and utilities to make working with OpenCV easier.
* [Dlib library](http://dlib.net/) - Implementations of state-of-the-art CV and ML algorithms (including face recognition).
* [scikit-learn library](https://scikit-learn.org/stable/) - Machine learning in Python. Simple. Efficient. Beautiful, easy to use API.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 


## References

* [Drowsiness detection with OpenCV](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
