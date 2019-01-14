# Drowsiness-Detector
Drowsiness Detector is a computer vision system that automatically detects if the user drowsiness in real-time from a live video stream and then alert the user with an alarm notification. 

This repository is based on the tutorial by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Install and set up Python 3.
1. Install cmake in your system

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
    python detect_drowsiness.py
    ```

## Built With

* [OpenCV Library](https://opencv.org/) - Most used computer vision library. Highly efficient. Facilitates real-time image processing.
* [imutils library](https://github.com/jrosebr1/imutils) -  A collection of helper functions and utilities to make working with OpenCV easier.
* [Dlib library](http://dlib.net/) - Implementations of state-of-the-art CV and ML algorithms (including face recognition).
* [scikit-learn library](https://scikit-learn.org/stable/) - Machine learning in Python. Simple. Efficient. Beautiful, easy to use API.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 


## References

* [Drowsiness detection with OpenCV](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
