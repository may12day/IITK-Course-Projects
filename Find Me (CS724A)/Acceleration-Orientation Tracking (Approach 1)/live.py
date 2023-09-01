import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pykalman import KalmanFilter

from scipy import integrate

ax_new = []
ay_new = []
az_new = []
data_smoothened = []

df = pd.read_csv('sensor.csv',sep=',', header=None)
df = df.iloc[:,4:6]

def predict(df):
    global data_smoothened
    if(df.shape[0] == len(data_smoothened)):
        return False
    data = df.to_numpy()
    ism = [data[0, 0],0,data[0, 1],0]
    tm = [[1, 1, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1],[0, 0, 0, 1]]
    om = [[1, 0, 0, 0],[0, 0, 1, 0]]
    kf1 = KalmanFilter(transition_matrices = tm,observation_matrices = om,initial_state_mean = ism)
    kf1 = kf1.em(data, n_iter=5)
    (smoothed_state_means, _ ) = kf1.smooth(data)
    plt.plot(smoothed_state_means[:, 0], smoothed_state_means[:, 2], 'r--')
    # data_smoothened = smoothed_state_means 
    # return True
    return smoothed_state_means[:, 0], smoothed_state_means[:, 2]


def animate(i):
    data = pd.read_csv('sensor.csv', header=None)

    t = data[0]
    t1 = data[2]
    t2 = data[3]
    t3 = data[1]
    arr = predict(df)
    ax = arr[0]
    ay = arr[1]
    az = data[6]


    R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    R[0][0] = np.cos(t2[t2.size-1]*(np.pi/180))*np.cos(t3[t3.size-1]*(np.pi/180))
    R[0][1] = np.sin(t1[t1.size-1]*(np.pi/180))*np.sin(t2[t2.size-1]*(np.pi/180))*np.cos(t3[t3.size-1]*(np.pi/180)) - np.cos(t1[t1.size-1]*(np.pi/180))*np.sin(t3[t3.size-1]*(np.pi/180))
    R[0][2] = np.cos(t1[t1.size-1]*(np.pi/180))*np.sin(t2[t2.size-1]*(np.pi/180))*np.cos(t3[t3.size-1]*(np.pi/180)) + np.sin(t1[t1.size-1]*(np.pi/180))*np.cos(t3[t3.size-1]*(np.pi/180))
    R[1][0] = np.cos(t2[t2.size-1]*(np.pi/180))*np.sin(t3[t3.size-1]*(np.pi/180))
    R[1][1] = np.sin(t1[t1.size-1]*(np.pi/180))*np.sin(t2[t2.size-1]*(np.pi/180))*np.sin(t3[t3.size-1]*(np.pi/180)) + np.cos(t1[t1.size-1]*(np.pi/180))*np.cos(t3[t3.size-1]*(np.pi/180))
    R[1][2] = np.cos(t1[t1.size-1]*(np.pi/180))*np.sin(t2[t2.size-1]*(np.pi/180))*np.sin(t3[t3.size-1]*(np.pi/180)) - np.sin(t1[t1.size-1]*(np.pi/180))*np.cos(t3[t3.size-1]*(np.pi/180))
    R[2][0] = -np.sin(t2[t2.size-1]*(np.pi/180))
    R[2][1] = np.sin(t1[t1.size-1]*(np.pi/180))*np.cos(t2[t2.size-1]*(np.pi/180))
    R[2][2] = np.cos(t1[t1.size-1]*(np.pi/180))*np.cos(t2[t2.size-1]*(np.pi/180))

    acc = [ax[ax.size-1],ay[ay.size-1],az[az.size-1]]

    acc3d = np.matmul(R, acc)
    # acc3d = np.matmul(np.transpose(R), acc)
    ax_new.append(acc3d[0])
    ay_new.append(acc3d[1])
    az_new.append(acc3d[2])


    vx = [0]
    vy = [0]
    vz = [0]

    for i in range(len(ax_new)-1): 
        vx = vx + [vx[-1] + ax_new[i]*0.1]
        vy = vy + [vy[-1] + ay_new[i]*0.1]
        vz = vz + [vz[-1] + az_new[i]*0.1]
    
    x = [0]
    y = [0]
    z = [0]

    for i in range(len(ax_new)-1): 
        x = x + [x[-1] + vx[i]*0.1]
        y = y + [y[-1] + vy[i]*0.1]
        z = z + [z[-1] + vz[i]*0.1]

    plt.cla()
    plt.grid()
    
    img = plt.imread("boundary.jpg")
    plt.imshow(img, extent=[-5, 5, 5, -5])

    plt.plot(x,y, label='xy')
    if (x[-1]*x[-1] + y[-1]*y[-1] > 25):
        plt.title("Outside")
        # print("Outside")

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)   

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
