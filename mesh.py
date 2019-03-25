import numpy as np
from matplotlib import pyplot as plt
import math
import json
import mesh_function

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

# #####################Mode########################
# Mode1 : Horizon
# Mode2 : Vertical
# Mode3 : Complex
Mode = 1
# #####################Mode Settings###############
# Number of slices
N = 5
# #####################Sample######################
# line_type
seg_type = np.array([0, 0, 0, 0, 0])

# Circle center
center = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

# Circle radius
radius = np.array([0, 0, 0, 0, 0])

# Angle
angle = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

# ###################Cube Default####################
vertex0 = [0, 0]
vertex1 = [0, 55]
vertex2 = [55, 55]
vertex3 = [55, 0]
vertex4 = [0, 0]
vertex = np.array([[vertex0[0], vertex0[1]],
                   [vertex1[0], vertex1[1]],
                   [vertex2[0], vertex2[1]],
                   [vertex3[0], vertex3[1]],
                   [vertex4[0], vertex4[1]]
                   ])

# Create rect
t = np.arange(0, 1 + 1e-4, 1e-4)
part = mesh_function.CreateCurve(vertex, seg_type, center, radius, angle, t)

# Fill Part
dx = 0.05
dz = 0.05

# Rect Range
rect = np.array([-10, -10, 75, 85])
jzfb = mesh_function.FillPartGeometry(part, rect, dx, dz)

# ####################Transmit Probe###################
# [Number, pitch, width, start_X, start_Y, angle]
Tprobe = np.array([64, 0.4, 0.3, 14.5, 65, 0])

Transducer = mesh_function.LocateTransducerPosition(Tprobe, rect, dx, dz)
TransducerX = Transducer[0]
TransducerZ = Transducer[1]

# ####################Wedge Part#######################
seg_type_wedge = np.array([0, 0, 0, 0, 0])

center_wedge = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

radius_wedge = np.array([0, 0, 0, 0, 0])

angle_wedge = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

vertex_wedge0 = np.array([12.5, 55])
vertex_wedge1 = np.array([12.5, 65])
vertex_wedge2 = np.array([42.5, 65])
vertex_wedge3 = np.array([42.5, 55])
vertex_wedge4 = np.array([12.5, 55])

vertex_wedge = np.array([vertex_wedge0, vertex_wedge1, vertex_wedge2, vertex_wedge3, vertex_wedge4])

wedge = mesh_function.CreateCurve(vertex_wedge, seg_type_wedge, center_wedge, radius_wedge, angle_wedge, t)

jzfb_wedge = mesh_function.FillWedgeGeometry(wedge, rect, dx, dz)

# ######################defect part####################
defect = np.array([1, 27.5, 5, 0.5, 0.5, 0])
jzfb_defect_mesh = mesh_function.FillDefectGeometry(defect, rect, dx, dz)
jzfb_defect = jzfb_defect_mesh[0]
geometry = jzfb_defect_mesh[1]

# ######################Mesh part######################
# Horizon
interval = round(vertex1[1] / N, 2)
if Mode == 1:
    vertex_0 = np.array([vertex0[0], vertex0[1]])
    vertex_3 = np.array([vertex3[0], vertex3[1]])
    vertex_4 = vertex_0
    for k in range(N - 1):
        vertex_1 = np.array([vertex_0[0], vertex_0[1] + interval])
        vertex_2 = np.array([vertex_3[0], vertex_3[1] + interval])
        vertex_t = np.array([vertex_0, vertex_1, vertex_2, vertex_3, vertex_4])
        part_t = mesh_function.CreateCurve(vertex_t, seg_type, center, radius, angle, t)
        jzfb_t = mesh_function.FillPartGeometry(part_t, rect, dx, dz)
        index_t = np.nonzero(jzfb_t == 1)
        for j in range(index_t[0].shape[0]):
            jzfb[index_t[0][j], index_t[1][j]] = k + 1
        vertex_0 = np.array([vertex_1[0], vertex_1[1] - dz])
        vertex_3 = np.array([vertex_2[0], vertex_2[1] - dx])
        vertex_4 = vertex_0
    vertex_1 = vertex1
    vertex_2 = vertex2
    vertex_t = np.array([vertex_0, vertex_1, vertex_2, vertex_3, vertex_4])
    part_t = mesh_function.CreateCurve(vertex_t, seg_type, center, radius, angle, t)
    jzfb_t = mesh_function.FillPartGeometry(part_t, rect, dx, dz)
    index_t = np.nonzero(jzfb_t == 1)
    for j in range(index_t[0].shape[0]):
        jzfb[index_t[0][j], index_t[1][j]] = N
index_wedge = np.nonzero(jzfb_wedge == 2)
for i in range(index_wedge[0].shape[0]):
    jzfb[index_wedge[0][i], index_wedge[1][i]] = 0

index_defect = np.nonzero(jzfb_defect == -2)
for i in range(index_defect[0].shape[0]):
    jzfb[index_defect[0][i], index_defect[1][i]] = -2

# 信号形式
sig = np.array([0, 5e6, 3])

# 介质1参数
jz1 = np.array([1000, 1496.6, 0])

# 介质2参数
jz2 = np.array([2700, 6300, 3080])

# 空间间隔
fdtdpara0 = 0.05e-3

# 时间间隔
fdtdpara1 = 0.005e-6

# 时长
fdtdpara2 = 50e-6

# 换能器阵元表面角度
fdtdpara3 = 90 - Tprobe[5]

# 阵元大小
fdtdpara4 = Tprobe[2]
fdtdpara = np.array([fdtdpara0, fdtdpara1, fdtdpara2, fdtdpara3, fdtdpara4])

mesh_function.SaveFDTDMesh('mesh', jzfb, TransducerX, TransducerZ, TransducerX, TransducerZ, part, wedge, defect, geometry, rect, sig, jz1, jz2, fdtdpara, dx, dz)
jzfb_Tranpose = np.squeeze(flip90_left(jzfb))
plt.imshow(jzfb_Tranpose)
plt.show()


'''
    #仿真区域点数
    Nx = jzfb.shape[0]
    Nz = jzfb.shape[1]
    #计算换能器阵元的网格索引
    NElement = TTransducerX.shape[0]
    Na = TTransducerX.shape[1]
    #工件外轮廓点数
    Np = part.shape[0]
    #楔快外轮廓点数
    Nw = wedge.shape[0]
    Nd = geometry.shape[0]

    pointnum = [Nx, Nz, Np, NElement, Na, Nw, Nd]
    rect = rect.tolist()
    sig = sig.tolist()
    jz1 = jz1.tolist()
    jz2 = jz2.tolist()
    fdtdpara = fdtdpara.tolist()
    dxz = [dx, dz]
    jzfb = jzfb.tolist()
    part = part.tolist()
    TTransducerX = TTransducerX.tolist()
    TTransducerZ =TTransducerZ.tolist()
    RTransducerX = RTransducerX.tolist()
    RTransducerZ = RTransducerZ.tolist()
    wedge = wedge.tolist()
    geometry = geometry.tolist()

    data_dict = {'pointnum': pointnum, 'rect': rect, 'sig': sig, 'jz1': jz1, 'jz2': jz2,
                 'fdtdpara': fdtdpara, 'dxz': dxz, 'jzfb': jzfb, 'part': part,
                 'TTransducerX': TTransducerX, 'TTransducerZ': TTransducerZ,
                 'RTransducerX': RTransducerX, 'RTransducerZ': RTransducerZ,
                 'wedge': wedge, 'geometry': geometry}
    data_json = json.dumps(data_dict)
    data_str = str(data_json)
    with open(filename + '.json', 'w') as fid:
        #json.dump(data_json, fid)
        fid.write(data_str)
    fid.close()
'''