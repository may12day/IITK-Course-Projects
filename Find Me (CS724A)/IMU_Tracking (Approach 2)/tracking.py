from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot

import numpy
import pandas as pd
import os
#import time
os.system('python demo.py')
#time.sleep(5)
data = numpy.genfromtxt("sensor.csv", delimiter=",", skip_header=1)

sample_rate = 400

timestamp = (data[:, 0])/1000
accelerometer = data[:, 1:4]
gyroscope = data[:, 4:7]


# Intantiate AHRS algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(0.5,  # gain
                                   10,  # acceleration rejection
                                   0,  # magnetic rejection
                                   5 * sample_rate)  # rejection timeout = 2 seconds



# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])
    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])
    euler[index] = ahrs.quaternion.to_euler()
    acceleration[index] = ahrs.earth_acceleration


_, axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})

axes[0].plot(timestamp, acceleration[:, 0], "tab:red", label="X")
axes[0].plot(timestamp, acceleration[:, 1], "tab:green", label="Y")
axes[0].plot(timestamp, acceleration[:, 2], "tab:blue", label="Z")
axes[0].set_title("Acceleration")
axes[0].set_ylabel("m/s/s")
axes[0].grid()
axes[0].legend()

# Identify moving periods
is_moving = numpy.empty(len(timestamp))

for index in range(len(timestamp)):
    is_moving[index] = numpy.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s

margin = int(0.1 * sample_rate)  # 100 ms

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

# Plot moving periods
axes[1].plot(timestamp, is_moving, "tab:cyan", label="Is moving")
pyplot.sca(axes[1])
pyplot.yticks([0, 1], ["False", "True"])
axes[1].grid()
axes[1].legend()


velocity = numpy.zeros((len(timestamp), 3))
for index in range(len(timestamp)):
    if is_moving[index]:  # only integrate if moving
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]



# Find start and stop indices of each moving period
is_moving_diff = numpy.diff(is_moving, append=is_moving[-1])

@dataclass
class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1


is_moving_periods = []
is_moving_period = IsMovingPeriod()

for index in range(len(timestamp)):
    if is_moving_period.start_index == -1:
        if is_moving_diff[index] == 1:
            is_moving_period.start_index = index

    elif is_moving_period.stop_index == -1:
        if is_moving_diff[index] == -1:
            is_moving_period.stop_index = index
            is_moving_periods.append(is_moving_period)
            is_moving_period = IsMovingPeriod()


# Remove integral drift from velocity
velocity_drift = numpy.zeros((len(timestamp), 3))

for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index

    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]

    t_new = timestamp[start_index:(stop_index + 1)]

    velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
    velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
    velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

velocity = velocity - velocity_drift

# Plot velocity
axes[2].plot(timestamp, velocity[:, 0], "tab:red", label="X")
axes[2].plot(timestamp, velocity[:, 1], "tab:green", label="Y")
axes[2].plot(timestamp, velocity[:, 2], "tab:blue", label="Z")
axes[2].set_title("Velocity")
axes[2].set_ylabel("m/s")
axes[2].grid()
axes[2].legend()


# Calculate position
position = numpy.zeros((len(timestamp), 3))
for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

# Plot position
axes[3].plot(timestamp, position[:, 0], "tab:red", label="X")
axes[3].plot(timestamp, position[:, 1], "tab:green", label="Y")
axes[3].plot(timestamp, position[:, 2], "tab:blue", label="Z")
axes[3].set_title("Position")
axes[3].set_xlabel("Seconds")
axes[3].set_ylabel("m")
axes[3].grid()
axes[3].legend()


# Create 3D animation (takes a long time, set to False to skip)
if True:
    figure = pyplot.figure(figsize=(10, 10))

    #axes = pyplot.axes(projection="3d")
    #axes.set_xlabel("m")
    #axes.set_ylabel("m")
    #axes.set_zlabel("m")

    x = []
    y = []
    z = []

    #scatter = axes.scatter(x, y, z)

    fps = 30
    samples_per_frame = int(sample_rate / fps)

    def update(frame):

        

        index = frame * samples_per_frame

        #axes.set_title("{:.3f}".format(timestamp[index]) + "X_Axis")

        x.append(position[index, 0])
        y.append(position[index, 1])
        z.append(position[index, 2])

        print(position[index, 0],position[index, 1])

        #scatter._offsets3d = (x, y, z)

        pyplot.cla()
        pyplot.grid()

        f = open("output.csv",mode="a")
        f.write(str(x)+","+str(y)+","+str(z)+"\n")
        f.close()


        #pyplot.plot3D (x,y,z,'red')
        pyplot.plot(z,x, label='xy')
        if (x[-1]*x[-1] + y[-1]*y[-1] > 10000):
            print("Outside")

        pyplot.xlim(-400,400)
        pyplot.ylim(-400,400)   

        pyplot.legend(loc='upper left')
        pyplot.tight_layout()

        # if (min(x) != max(x)) and (min(y) != max(y)) and (min(z) != max(z)):
        #     axes.set_xlim3d(min(x), max(x))
        #     axes.set_ylim3d(min(y), max(y))
        #     axes.set_zlim3d(min(z), max(z))

        #     axes.set_box_aspect((numpy.ptp(x), numpy.ptp(y), numpy.ptp(z)))

        # return scatter

    anim = animation.FuncAnimation(figure, update,
                                   frames=int(len(timestamp) / samples_per_frame),
                                   interval=100 / fps,
                                   repeat=False)

    anim.save("animation.gif", writer=animation.PillowWriter(fps))

pyplot.show()
