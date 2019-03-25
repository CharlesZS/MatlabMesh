import numpy as np
import math
import struct
import json


# ####################生成边缘曲线##########################
def CreateCurve(vertex, seg_type, center, radius, angle, t):
    Nv = (vertex.shape)[0]
    total = -1
    num = t.shape[0]
    geometry = np.zeros(shape=((Nv - 1) * num ,2))
    
    for i in range(Nv - 1):
        if(seg_type[i] == 0):
            p1 = np.squeeze(vertex[np.mod(i, Nv),])
            p2 = np.squeeze(vertex[np.mod(i + 1, Nv), ])
            linex = p1[0] * (1- t) + p2[0] * t
            liney = p1[1] * (1- t) + p2[1] * t
            geometry[total + 1 : total + num + 1, 0] = linex[:]
            geometry[total + 1 : total + num + 1, 1] = liney[:]
            total = total + num
            
        elif(seg_type[i] == 1):
            angle_range = (agnle[i, 0] * (1 - t) + angle[i, 1] * t)/180 * np.pi
            arcx = center[i, 0] + radius[i] * np.cos(angle_range)
            arcy = center[i, 0] + radius[i] * np.sin(angle_range)
            geometry[total + 1 : total + num + 1, 0] = arcx[:]
            geometry[total + 1 : total + num + 1, 1] = acry[:]
            total = total + num

    return geometry
            

##################生成试件区域的网格#########################
def FillPartGeometry(part, rect, dx, dz):
    x0 = rect[0]
    z0 = rect[1]
    w = rect[2]
    h = rect[3]
    Nx = int(np.round(w / dx) + 1)
    Nz = int(np.round(h / dz) + 1)
    #x字典的方向
    dict_num = np.zeros((Nx, Nz))   ##不止3层 根据工件的复杂度递增
    
    dict_count = np.zeros((Nx, 1))
    jzfb = -1 * np.ones((Nx, Nz))

    #第一个点 python的索引从零开始 所以减一
    indexX = int(np.round((part[0, 0] - x0) / dx) )
    indexZ = int(np.round((part[0, 1] - z0) / dz) )
    prev = indexX
    dict_count[indexX] = dict_count[indexX] + 1
    dict_num[indexX, int(dict_count[indexX])] = indexZ

    
    for i in range(1, part.shape[0]):
        indexX = int(np.round((part[i, 0] - x0) / dx))
        indexZ = int(np.round((part[i, 1] - z0) / dz))
        if prev != indexX:
            dict_count[indexX] = dict_count[indexX] + 1
            dict_num[indexX, int(dict_count[indexX])] = indexZ
            prev = indexX

            
    #删除重复的
    for i in range(Nx):
        temp = np.squeeze(dict_num[i, range(0,int(dict_count[i] + 1))])
        temp = np.unique(temp)
        for j in range(temp.shape[0]):
            dict_num[i, j] = temp[j]
        LL = dict_num.shape[1]
        for j in range(temp.shape[0], LL):
            dict_num[i, j] = 0
        dict_count[i] = temp.shape[0]
    

    #开始填充
    for i in range(Nx):
        flag = -1
        for j in range(int(dict_count[i]) - 1):
            if dict_num[i, j] < dict_num[i, j + 1]:
                jzfb[i, range(int(dict_num[i, j]), int(dict_num[i, j + 1]) + 1)] = flag
            else:
                jzfb[i, range(int(dict_num[i, j]), int(dict_num[i, j + 1]) + 1, -1)] = flag
            flag = -1 * flag
            
    jzfb = jzfb.astype(int)
    
    return jzfb

def LocateTransducerPosition(probe, rect, dx, dz):
    N = int(probe[0])    #阵元数量
    pitch = probe[1]    #阵元间距
    a = probe[2]    #阵元宽度
    WedgeStartX = probe[3]  
    WedgeStartZ = probe[4]
    WedgeAngle = probe[5] 

    x0 = rect[0];
    z0 = rect[1];
    w = rect[2];
    h = rect[3];

    element = np.arange(0, a + dx, dx)
    Xt = np.zeros((N, element.shape[0]))
    Zt = np.zeros((N, element.shape[0]))

    for i in range(N):
        Xt[i, :] = WedgeStartX + (i * pitch + element) * math.cos(WedgeAngle)
        Zt[i, :] = WedgeStartZ + (i * pitch + element) * math.sin(WedgeAngle)

    TransducerX = np.round((Xt - x0) / dx)
    TransducerZ = np.round((z0 + h - Zt) / dz) + 1
    TransducerX = TransducerX.astype(int)
    TransducerZ = TransducerZ.astype(int)
    Transducer = np.array([TransducerX, TransducerZ])
    
    return Transducer



##################生成楔块区域的网格#########################
def FillWedgeGeometry(wedge, rect, dx, dz):
    x0 = rect[0]
    z0 = rect[1]
    w = rect[2]
    h = rect[3]
    Nx = int(np.round(w / dx) + 1)
    Nz = int(np.round(h / dz) + 1)
    #x字典的方向
    dict_num = np.zeros((Nx, Nz))   ##不止3层 根据工件的复杂度递增
    
    dict_count = np.zeros((Nx, 1))
    jzfb = np.zeros((Nx, Nz))

    #第一个点 python的索引从零开始 所以减一
    indexX = int(np.round((wedge[0, 0] - x0) / dx) )
    indexZ = int(np.round((wedge[0, 1] - z0) / dz) )
    prev = indexX
    dict_count[indexX] = dict_count[indexX] + 1
    dict_num[indexX, int(dict_count[indexX])] = indexZ

    
    for i in range(1, wedge.shape[0]):
        indexX = int(np.round((wedge[i, 0] - x0) / dx))
        indexZ = int(np.round((wedge[i, 1] - z0) / dz))
        if prev != indexX:
            dict_count[indexX] = dict_count[indexX] + 1
            dict_num[indexX, int(dict_count[indexX])] = indexZ
            prev = indexX

            
    #删除重复的
    for i in range(Nx):
        temp = np.squeeze(dict_num[i, range(0,int(dict_count[i] + 1))])
        temp = np.unique(temp)
        for j in range(temp.shape[0]):
            dict_num[i, j] = temp[j]
        LL = dict_num.shape[1]
        for j in range(temp.shape[0], LL):
            dict_num[i, j] = 0
        dict_count[i] = temp.shape[0]



    #开始填充
    for i in range(Nx):
        if dict_count[i] ==3:
            if dict_num[i, 1] < dict_num[i, 2]:
                jzfb[i, range(int(dict_num[i, 1]), int(dict_num[i, 2]) + 1)] = 2
            else:
                jzfb[i, range(int(dict_num[i, 1]), int(dict_num[i, 2]) + 1, -1)] = 2
            
    jzfb = jzfb.astype(int)
    
    return jzfb



##################生成缺陷区域的网格#########################
def FillDefectGeometry(defect, rect, dx, dz):
    x0 = rect[0]
    z0 = rect[1]
    w = rect[2]
    h = rect[3]
    Nx = int(np.round((w / dx) + 1))
    Nz = int(np.round((h / dz) + 1))

    jzfb = np.zeros((Nx, Nz))

    t = np.arange(0, 1 + 1e-3, 1e-3)
    geometry = np.zeros((t.shape[0], 2))
    if defect[0] == 1:
        for i in range(t.shape[0]):
            geometry[i, 0] = defect[1] + defect[3] * math.cos(2 * math.pi * t[i])
            geometry[i, 1] = defect[2] + defect[3] * math.sin(2 * math.pi * t[i])
    elif defect[0] == 2:
        xright = defect[3] / 2
        ztop = defect[4] / 2
        theta = defect[5] / 180 * math.pi
        RotateM = np.array([[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        p0 =  np.dot(RotateM, np.array([xright, ztop]))
        p1 =  np.dot(RotateM, np.array([-xright, ztop]))
        p2 =  np.dot(RotateM, np.array([-xright, -ztop]))
        p3 =  np.dot(RotateM, np.array([xright, -ztop]))
        
        seg_type0 = 0
        seg_type1 = 0
        seg_type2 = 0
        seg_type3 = 0
        seg_type4 = 0
        seg_type = np.array([seg_type0,seg_type1,seg_type2,seg_type3,seg_type4])

        center0 = [0,0]
        center1 = [0,0]
        center2 = [0,0]
        center3 = [0,0]
        center4 = [0,0]
        center = np.array([[center0[0], center0[1]],
                           [center1[0], center1[1]],
                           [center2[0], center2[1]],
                           [center3[0], center3[1]],
                           [center4[0], center4[1]]
                           ])

        radius0 = 0
        radius1 = 0
        radius2 = 0
        radius3 = 0
        radius4 = 0
        radius = np.array([radius0, radius1, radius2, radius3, radius4])

        angle0 = [0, 0]
        angle1 = [0, 0]
        angle2 = [0, 0]
        angle3 = [0, 0]
        angle4 = [0, 0]
        angle = np.array([[angle0[0], angle0[1]],
                          [angle1[0], angle1[1]],
                          [angle2[0], angle2[1]],
                          [angle3[0], angle3[1]],
                          [angle4[0], angle4[1]],
                        ])

        vertex0 = np.array([defect[1], defect[2]]) + p0
        vertex1 = np.array([defect[1], defect[2]]) + p1
        vertex2 = np.array([defect[1], defect[2]]) + p2
        vertex3 = np.array([defect[1], defect[2]]) + p3
        vertex4 = vertex0
        vertex = np.array([vertex0, vertex1, vertex2, vertex3, vertex4])

        t = np.arange(0, 1 + 1e-3, 1e-3)

        geometry = CreateCurve(vertex, seg_type, center, radius, angle, t)


    dict_num = np.zeros((Nx, 4))   ##不止3层 根据工件的复杂度递增
    
    dict_count = np.zeros((Nx, 1))

    #第一个点 python的索引从零开始 所以减一
    indexX = int(np.round((geometry[0, 0] - x0) / dx) )
    indexZ = int(np.round((geometry[0, 1] - z0) / dz) )
    prev = indexX
    dict_count[indexX] = dict_count[indexX] + 1
    dict_num[indexX, int(dict_count[indexX])] = indexZ

    
    for i in range(1, geometry.shape[0]):
        indexX = int(np.round((geometry[i, 0] - x0) / dx))
        indexZ = int(np.round((geometry[i, 1] - z0) / dz))
        if prev != indexX:
            dict_count[indexX] = dict_count[indexX] + 1
            dict_num[indexX, int(dict_count[indexX])] = indexZ
            prev = indexX

            
    #删除重复的
    for i in range(Nx):
        temp = np.squeeze(dict_num[i, range(0,int(dict_count[i] + 1))])
        temp = np.unique(temp)
        for j in range(temp.shape[0]):
            dict_num[i, j] = temp[j]
        LL = dict_num.shape[1]
        for j in range(temp.shape[0], LL):
            dict_num[i, j] = 0
        dict_count[i] = temp.shape[0]



    #开始填充
    for i in range(Nx):
        if dict_count[i] ==3:
            if dict_num[i, 1] < dict_num[i, 2]:
                jzfb[i, range(int(dict_num[i, 1]), int(dict_num[i, 2]) + 1)] = -2
            else:
                jzfb[i, range(int(dict_num[i, 1]), int(dict_num[i, 2]) + 1, -1)] = -2
            
    jzfb = jzfb.astype(int)

    jzfb_plus = [jzfb, geometry]
    
    return jzfb_plus 



##############保存网格#######################
def SaveFDTDMesh(filename, jzfb, TTransducerX, TTransducerZ, RTransducerX, RTransducerZ, part, wedge, defect, geometry, rect, sig, jz1, jz2, fdtdpara, dx, dz):
    fid = open(filename + ".mesh", "wb")
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

    pointnum = np.array([Nx, Nz, Np, NElement, Na, Nw, Nd])

    for i in range(pointnum.shape[0]):
        byte = struct.pack('i', pointnum[i])
        fid.write(byte)

    for i in range(rect.shape[0]):
        byte = struct.pack('d', rect[i])
        fid.write(byte)

    for i in range(sig.shape[0]):
        byte = struct.pack('d', sig[i])
        fid.write(byte)

    for i in range(jz1.shape[0]):
        byte = struct.pack('d', jz1[i])
        fid.write(byte)

    for i in range(jz2.shape[0]):
        byte = struct.pack('d', jz2[i])
        fid.write(byte)

    for i in range(fdtdpara.shape[0]):
        byte = struct.pack('d', fdtdpara[i])
        fid.write(byte)

    dxz = np.array([dx, dz])

    for i in range(dxz.shape[0]):
        byte = struct.pack('d', dxz[i])
        fid.write(byte)

    for i in range(Nx):
        for j in range(Nz):
            byte = struct.pack('i', int(jzfb[i, j]))
            fid.write(byte)
            

    if Np != 0:
        for i in range(part.shape[0]):
            byte = struct.pack('d', part[i, 0])
            fid.write(byte)
        for i in range(part.shape[0]):
            byte = struct.pack('d', part[i, 1])
            fid.write(byte)

    if NElement != 0:
        for i in range(NElement):
            for j in range(TTransducerX.shape[1]):
                byte = struct.pack('d', TTransducerX[i, j])
                fid.write(byte)

            for j in range(TTransducerZ.shape[1]):
                byte = struct.pack('d', TTransducerZ[i, j])
                fid.write(byte)

            for j in range(RTransducerX.shape[1]):
                byte = struct.pack('d', RTransducerX[i, j])
                fid.write(byte)

            for j in range(RTransducerZ.shape[1]):
                byte = struct.pack('d', RTransducerZ[i, j])
                fid.write(byte)

    if Nw != 0:
        for i in range(wedge.shape[0]):
            byte = struct.pack('d', wedge[i, 0])
            fid.write(byte)
        for i in range(wedge.shape[0]):
            byte = struct.pack('d', wedge[i, 1])
            fid.write(byte)

    if Nd != 0:
        for i in range(geometry.shape[0]):
            byte = struct.pack('d', geometry[i, 0])
            fid.write(byte)
        for i in range(geometry.shape[0]):
            byte = struct.pack('d', geometry[i, 1])
            fid.write(byte)

    fid.close()

################保存为Json格式#################
    




































    







