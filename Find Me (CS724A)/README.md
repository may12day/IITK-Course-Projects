# Find Me
Real‑time Localization based on Sensing, Communication and Networking

## Steps to RUN Acceleration-Orientation Tracking (Approach 1):

1) Clear the ```sensor.csv``` before running to record fresh data.
2) Use [HyperIMU](https://play.google.com/store/apps/details?id=com.ianovir.hyper_imu) mobile app to stream sensor data in real-time. Go to its Settings, enter the Laptop's IP address in the app and update the sampling frequency to 1000 ms. Make sure the laptop and mobile phone is connected to the same network. Then go to Sensor List and toggle ON the Linear Accelerometer and Orientation buttons.
3) Click the button on the app's main interface to initiate streaming.
4) Run ```demo.py``` to start recording sensor data in ```sensor.csv```.
```sh
python3 demo.py
```
5) Then run ```live.py``` to get real-time position tracking on the XY plane. 
```sh
python3 live.py
```

## Steps to RUN IMU_Tracking (Approach 2):

1) Clear the ```sensor.csv``` before running to record fresh data.
2) Use [HyperIMU](https://play.google.com/store/apps/details?id=com.ianovir.hyper_imu) mobile app to stream sensor data in real-time. Go to its Settings, enter the Laptop's IP address in the app and update the sampling frequency to 1000 ms. Make sure the laptop and mobile phone is connected to the same network. Then go to Sensor List and toggle ON the Linear Accelerometer and Gyroscope buttons.
3) Click the button on the app's main interface to initiate streaming.
4) Run ```tracking.py``` to start recording sensor data in ```sensor.csv``` and display the path once travelling is completed.
```sh
python3 tracking.py
```
5) The animation of the path covered is saved as ```animation.gif```.
