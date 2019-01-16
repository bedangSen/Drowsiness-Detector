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
 
    ```
    # define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold for to set off the
	# alarm
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 30

	# initialize the frame counter as well as a boolean used to
	# indicate if the alarm is going off
	COUNTER = 0
	ALARM_ON = False
    ```
    
 1. The dlib library ships with a Histogram of Oriented Gradients-based face detector along with a facial landmark predictor — we instantiate both of these in the following code block:
 
    ```
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
    ```

 1. Therefore, to extract the eye regions from a set of facial landmarks, we simply need to know the correct array slice indexes:

    ```
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    ```

 1. Using these indexes, we’ll easily be able to extract the eye regions via an array slice. We are now ready to start the core of our drowsiness detector:

   ```
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
    ```

 1. The next step is to apply facial landmark detection to localize each of the important regions of the face:

    ```
# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
    ```

 1. We can then visualize each of the eye regions on our frame  by using the cv2.drawContours  function below — this is often helpful when we are trying to debug our script and want to ensure that the eyes are being correctly detected and localized:

    ```
# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    ```

 1. Finally, we are now ready to check to see if the person in our video stream is starting to show symptoms of drowsiness:

    ```
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False
    ```

 1. The final code block in our drowsiness detector handles displaying the output frame  to our screen:
 
    ```
		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
    ```

 1. Run the application.

    ```
    python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
    ```
    
## Built With

* [OpenCV Library](https://opencv.org/) - Most used computer vision library. Highly efficient. Facilitates real-time image processing.
* [imutils library](https://github.com/jrosebr1/imutils) -  A collection of helper functions and utilities to make working with OpenCV easier.
* [Dlib library](http://dlib.net/) - Implementations of state-of-the-art CV and ML algorithms (including face recognition).
* [scikit-learn library](https://scikit-learn.org/stable/) - Machine learning in Python. Simple. Efficient. Beautiful, easy to use API.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 


## References

* [Drowsiness detection with OpenCV](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
