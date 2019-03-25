import numpy as np
import math
import struct
from matplotlib import pyplot as plt
from matplotlib import animation
import solution_function
import time
#####################################

######################################



#############试件部分###############
seg_type0 = 0
seg_type1 = 0
seg_type = np.array([seg_type0,seg_type1])

center0 = [0,0]
center1 = [0,0]
center = np.array([[center0[0], center0[1]],
                   [center1[0], center1[1]]
                   ])

radius0 = 0
radius1 = 0
radius = np.array([radius0, radius1])

angle0 = [0, 0]
angle1 = [0, 0]
angle = np.array([[angle0[0], angle0[1]],
                  [angle1[0], angle1[1]]
                ])


vertex0 = [-35, 0]
vertex1 = [10, 0]
vertex = np.array([[vertex0[0], vertex0[1]],
                   [vertex1[0], vertex1[1]]                   
                   ])


t = np.arange(0, 1 + 1e-4, 1e-4)
interface = solution_function.CreateCurve(vertex, seg_type, center, radius, angle, t)

#####################换能器部分#######################
WedgeAngle = 26
WedgeStartX = -30
WedgeStartY = 5

probe0 = 32
probe1 = 0.6
probe2 = 0.5
probe3 = WedgeStartX
probe4 = WedgeStartY
probe5 = WedgeAngle * math.pi / 180

probe = np.array([probe0, probe1, probe2, probe3, probe4, probe5])

N = int(probe[0])
pitch = probe[1]
Xt = np.zeros((N))
Zt = np.zeros((N))

for i in range(N):
    Xt[i] = probe[3]+ (i * pitch) * math.cos(probe[5])
    Zt[i] = probe[4] + (i * pitch) * math.sin(probe[5])

ChannelNum = 30
TFocus = np.zeros((ChannelNum, 2))

for i in range(ChannelNum):
    TFocus[i, 0] = 0
    TFocus[i, 1] = -0.1 * (i + 1) - 3
    
RFcous = TFocus
ApertureLaw = np.zeros((ChannelNum, N))
for i in range(ChannelNum):
    ApertureLaw[i] = np.array([range(N)])

c1 = 1496.6
c2 = 3080

[DelayLaw, xpos, zpos] = solution_function.CreateFocalLaw(TFocus, ApertureLaw, interface, Xt, Zt, c1, c2)


###########################
TApertureStart = np.zeros(ChannelNum)
TApertureStart[:] = 1
TApertureEnd = np.zeros(ChannelNum)
TApertureEnd[:] = 32
RApertureStart = np.zeros(ChannelNum)
RApertureStart[:] = 1
RApertureEnd = np.zeros(ChannelNum)
RApertureEnd[:] = 32

TApertureLaw = np.zeros((ChannelNum, N))
RApertureLaw = np.zeros((ChannelNum, N))

Tfocus = np.zeros((ChannelNum, 2))

Txpos = np.zeros((ChannelNum, N))
Tzpos = np.zeros((ChannelNum, N))
Rxpos = np.zeros((ChannelNum, N))
Rzpos = np.zeros((ChannelNum, N))

fig = plt.figure()
window = fig.add_subplot(111)

fid = -1
line, = window.plot([], [])

times = [i for i in range(ChannelNum)]
    

def frame_animate(fid):
    
    line, = window.plot([], [])
    for i in range(int(TApertureStart[fid]) - 1, int(TApertureEnd[fid])):
        line.set_data([], [])
    Tfocus[fid] = np.array([0, TFocus[fid, 0]])
    for i in range(int(TApertureStart[fid]) - 1, int(TApertureEnd[fid])):
        x1 = Xt[i]
        z1 = Zt[i]
        x2 = xpos[fid, i]
        z2 = zpos[fid, i]
        x4 = 0
        z4 = -1 * (TFocus[fid, 1] + 6)
        z3 = -3
        x3 = (z3 - z2) * (0 - x2) / (TFocus[fid, 1] - z2) + x2
        Txpos[fid, i - int(TApertureStart[fid]) + 1] = x3
        Tzpos[fid, i - int(TApertureStart[fid]) + 1] = -3
        Tfocus[fid, 1] = z4

        x = [x1, x2, x3, x4] 
        z = [z1, z2, z3, z4]
        line, = window.plot(x, z)

    return line

#anim = animation.FuncAnimation(fig, frame_animate, times, interval=600)
#plt.show()

solution_function.SaveInspectSolution('soluion', DelayLaw, ApertureLaw, DelayLaw, ApertureLaw, Tfocus, Txpos, Tzpos, Tfocus, Txpos, Tzpos)
        












