import cv2
import numpy as np
import time

#imports to load trained model
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# imports for email
import smtplib
import imghdr
from email.message import EmailMessage

def preprocess_frame(img_rgb):
    #This function preprocesses the video frame to be compatible with resnet50 input
    resized = cv2.resize(img_rgb, (224, 224))
    img = preprocess_input(resized)
    img = img.reshape(1,224,224,3)
    return img

def send_mail2(frame):
    #This function sends a mail everytime its called.
    Sender_Email = "xxxx@iitk.ac.in"
    Reciever_Email = "xxxx@gmail.com"
    Password = "xxxx"

    cv2.imwrite("test1.png", frame)
    filename = "test1.png"  # In the same directory as the script

    newMessage = EmailMessage()
    newMessage['Subject'] = "Fire Detected"
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email
    newMessage.set_content('Fire is detected in the region which is shown in image')

    with open(filename, 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.cc.iitk.ac.in', 465) as smtp:
        smtp.login(Sender_Email, Password)
        smtp.send_message(newMessage)

model1 = keras.models.load_model('./models/BS4_SGD_Pat10') #This is the model used to classify video frame as fire/smoke/none

#vid = cv2.VideoCapture(1)
oldtime = 0
frame_count = 0
arealist = []
mail_sent = 0
arealist = [0, 0, 0]
while(True):
    # Capture the video frame by frame
    frame = cv2.imread('./test.png')
    ret = True
    #ret, frame = vid.read()
    if ret:
        frame_count += 1
        #send video frame to model
        if ((frame_count % 1) == 0):
            img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            img = preprocess_frame(img_rgb)
            y_pred = model1.predict(img)
            # prediction_confidence = [fire smoke]
            threshold = 0.6
            temp =[0, 0]
            if (y_pred[0][0]>= threshold):
                temp[0] = 1
            if (y_pred[0][1]>= threshold):
                temp[1] = 1
            if (temp[0] == 1): #fire
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) # convert frane to grayscale image
                ret, thresh = cv2.threshold(gray, 227, 255, cv2.THRESH_BINARY) #Perform binary thresholding on the grayscale image
                contours,_ = cv2.findContours(thresh, 1, 1) #generate contours on the binary image
                if len(contours) > 0:
                    c = max(contours, key = cv2.contourArea) #select the biggest contour for annotation
                    rect = cv2.minAreaRect(c) #generate a rectangle for the selected contour
                    (x,y),(w,h), a = rect #unpack rectangle dimensions (top right corner), (width, height), angle
                    box = cv2.boxPoints(rect) #generate a box from rect dimensions
                    box = np.int0(box) #convert all float coordinates to int for opencv
                    rect2 = cv2.drawContours(frame.copy(),[box],0,(0,0,255),10) #generate the video frame image with the contour indicating fire
                    rect2 = cv2.putText(rect2, 'Fire!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) #Add a text on the video frame
                    frame = rect2 #use this for live display
                    area = w * h #generate area from width and height of annotated fire boundaries
                    arealist.append(area) #append areas to arealist
                    if len(arealist) >4: #remove the 0th element from the arealist if it is more than 4 elements long
                        arealist.pop(0)
                    if len(arealist)>3:
                        avg = (arealist[0]+arealist[1]+arealist[2])*0.3333333 #generate the moving average of the previous three fire area samples
                        rate = np.true_divide((arealist[3] - avg), avg) #calculate rate of fire spread
                        if (time.time() - oldtime)>60: #send a mail every 60 seconds, user defined parameter
                            mail_sent = 0
                            if rate>0.133 and mail_sent==0:
                                oldtime = time.time()
                                send_mail2(frame)
                                mail_sent = 1
            if (temp[1] == 1): #smoke
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) # convert frane to grayscale image
                ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_TOZERO) #Perform binary thresholding on the grayscale image
                contours,_ = cv2.findContours(thresh, 1, 1) #generate contours on the binary image
                if len(contours) > 0:
                    c = max(contours, key = cv2.contourArea) #select the biggest contour for annotation
                    rect = cv2.minAreaRect(c) #generate a rectangle for the selected contour
                    (x,y),(w,h), a = rect #unpack rectangle dimensions (top right corner), (width, height), angle
                    box = cv2.boxPoints(rect) #generate a box from rect dimensions
                    box = np.int0(box) #convert all float coordinates to int for opencv
                    rect3 = cv2.drawContours(frame.copy(),[box],0,(0,0,255),10) #generate the video frame image with the contour indicating fire
                    frame = rect3 #use this for live display
        # Display the resulting frame
        title = "Live Video Feed"
        cv2.imshow(title, frame)
    else:
        break
    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
