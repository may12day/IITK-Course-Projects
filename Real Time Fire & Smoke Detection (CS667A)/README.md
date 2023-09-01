# CS667A
Course Project for CS667A (2022-23 I Semester): Introduction to Internet of Things and its Industrial Applications

Title:
Real-Time Fire & Smoke Detection

Overview:
    This project performs Real-Time Fire & Smoke Detection. This is the course project for CS667 by the Group IOT Kanpur.
    The project has three components:
    a. ML model based on Resnet50 to label video frames as fire,smoke or none.
    b. Localization module based on OpenCV to locate a source of fire in the video frame once it has been labeled as fire/fire+smoke.
    c. Hazardous/Non-hazardous fire detection
    d. Alert generation if hazardous fire detected
	
Note: The instructions given in this file have been given with respect to running this project on a Linux based environment.
	
How to Download the Machine Learning Models
	Please download and extract the Machine Learning model we have used from the following URL:
	https://drive.google.com/file/d/1VEWovt9yXY0oGtycTag5Pjd6IEvAy255/view
	
How to Install the Project:
    Give following command to run Makefile: make
    This will automatically download the dependencies listed in requirements.txt
	To run the code, give the following command: make run
	
Configuration:
    In the iotproj1.py, please edit the send_mail2(frame) function to configure the sender and receiver email and sender password. The sender mail needs to be an IITK mail account.

How to Run the Project:

    To access the code for ML model training, please use the Models.csv file. The Kaggle URL for each model can be used to access the model training code. Also, it brings out the accuracy of the various models.

    To run this program, navigate to the directory containing the code and the model. Then run
        ipython
    In the ipython shell, use the following command to execute the file:
        run iotproj1.py

    If getting an error:
        global /io/opencv/modules/videoio/src/cap_v4l.cpp (902) open VIDEOIO(V4L2:/dev/video0): can't open camera by index
    please change the number inside the parantheses on line 50: vid = cv2.VideoCapture(1) to to 0 or -1.

    After the model is loaded, a new window showing the webcam video will popup.
    If any fire is detected in the video input, the video stream shows the text "Fire!" and a bounding box containing the fire.
    The video frame labeled fire is then passed on to the OpenCV module to check if the rate of increase of fire is hazardous or not. If yes, an email alert is sent with the video frame image containing localised fire. This email is sent every 60 seconds.

    Another version of this program has been created for testing with static image in case of no webcam access. To run this, use the following command to execute:
        run iotproj2.py
    This code takes a saved image 'test.png' and feeds to the program repeatedly. The variables have been initialised to generate an email alert to facilitate testing.

    To quit the program, press 'q'.



Training Data:
    The data at the following URL has been used to train the model:
        https://drive.google.com/file/d/11Y54cYEJ-LkD8VST-h0P9KXqnwiYEEqU/view?usp=sharing
