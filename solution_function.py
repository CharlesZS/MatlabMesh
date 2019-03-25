import numpy as np
import math
import struct
import json
from matplotlib import animation


#####################生成边缘曲线##########################
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


###############生成延时法则###################
def CreateFocalLaw(focus, ApertureLaw, interface, xt, zt, c1, c2):
    Nf = ApertureLaw.shape[0]
    N = ApertureLaw.shape[1]

    delay = np.zeros((Nf, N))
    xpos = np.zeros((Nf, N))
    zpos = np.zeros((Nf, N))

    for j in range(Nf):
        xf = focus[j, 0]
        
        zf = focus[j, 1]
        
        count = 0
        for i in ApertureLaw[j]:
            i = int(i)
            TravelTime = np.sqrt((xt[i] - interface[:, 0]) ** 2 + (zt[i] - interface[:, 1])** 2)  / c1 + np.sqrt( (xf - interface[:, 0]) ** 2 + (zf - interface[:, 1]) ** 2) / c2
            delay[j, count] = np.min(TravelTime)
            
            index = np.argmin(TravelTime)
            xpos[j, count] = interface[index, 0]
            zpos[j, count] = interface[index, 1]
            count = count + 1


        delay[j] = np.max(delay[j]) - delay[j]

    delay = delay * 1e-3
    result = [delay, xpos, zpos]          
    return result

#####################保存延时法则文件###################
def SaveInspectSolution(filename, TDelayLaw, TApertureLaw, RDelayLaw, RApertureLaw, Tfocus, Txpos, Tzpos, Rfocus, Rxpos, Rzpos):  
    (NChannel, TNa) = TDelayLaw.shape
    RNa = RDelayLaw.shape[1]

    fid = open("filename" + ".solution", "wb")


    DelayLawNum = [NChannel, TNa, RNa]
    TDelayLaw = TDelayLaw.tolist()
    TApertureLaw = TApertureLaw.tolist()
    Tfocus = Tfocus.tolist()
    Txpos = Txpos.tolist()
    Tzpos = Tzpos.tolist()
    RDelayLaw = RDelayLaw.tolist()
    RApertureLaw = RApertureLaw.tolist()
    Rfocus = Rfocus.tolist()
    Rxpos = Rxpos.tolist()
    Rzpos = Rzpos.tolist()
    data_dict = {"DelayLawNum": DelayLawNum, "TDelayLaw": TDelayLaw, "TApertureLaw": TApertureLaw,
                 "Tfocus": Tfocus, "Txpos": Txpos, "Tzpos": Tzpos,"RDelayLaw": RDelayLaw, "RApertureLaw": RApertureLaw,
                 "Rfocus": Rfocus, "Rxpos": Rxpos, "Rzpos": Rzpos}
    data_json = json.dumps(data_dict, indent = 4)
    data_str = str(data_json)
    with open(filename + ".json", "w") as fid:
        #json.dump(data_json, fid)
        fid.write(data_str)
    fid.close()
"""
    for i in range(DelayLawNum.shape[0]):
        byte = struct.pack("i", DelayLawNum[i])
        fid.write(byte)

    for i in range(NChannel):
        for j in range(TDelayLaw.shape[1]):
            byte = struct.pack("d", TDelayLaw[i, j])
            fid.write(byte)
        for j in range(TApertureLaw.shape[1]):
            byte = struct.pack("i", int(TApertureLaw[i, j]))
            fid.write(byte)
            
        byte = struct.pack("d", Tfocus[i, 0])
        fid.write(byte)
        byte = struct.pack("d", Tfocus[i, 1])
        fid.write(byte)

        for j in range(Txpos.shape[1]):
            byte = struct.pack("d", Txpos[i, j])
            fid.write(byte)
        for j in range(Tzpos.shape[1]):
            byte = struct.pack("d", Tzpos[i, j])
            fid.write(byte)


        for j in range(RDelayLaw.shape[1]):
            byte = struct.pack("d", RDelayLaw[i, j])
            fid.write(byte)
        for j in range(RApertureLaw.shape[1]):
            byte = struct.pack("i", int(RApertureLaw[i, j]))
            fid.write(byte)
            
        byte = struct.pack("d", Rfocus[i, 0])
        fid.write(byte)
        byte = struct.pack("d", Rfocus[i, 1])
        fid.write(byte)

        for j in range(Rxpos.shape[1]):
            byte = struct.pack("d", Rxpos[i, j])
            fid.write(byte)
        for j in range(Rzpos.shape[1]):
            byte = struct.pack("d", Rzpos[i, j])
            fid.write(byte)

    fid.close()
"""
