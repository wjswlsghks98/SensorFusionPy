from callogg.util.logreader import MultiLogIterator
from callogg.tabulator import Tabulator
from glob import glob
import scipy.io as sio
import numpy as np


def Processor(tb):
    """ 
    Process Data for Tabulator format input
    """
    # Lane Data
    Data = {} 
    Data['lane'] = {}

    # Longitudinal values
    x =  tb.capnp_df["modelV2"]["modelV2.laneLines.0.x"].to_numpy()
    Data['lane']['x'] = np.array(x[0])

    # Lateral values
    prev_num = len(tb.capnp_df["modelV2"]["modelV2.laneLines.0.y"].to_numpy()[0])
    n = tb.capnp_df["modelV2"]["modelV2.laneLines.0.y"].to_numpy().size

    prev_num2 = 11
    x_ = np.linspace(0,100,11) # equal spacing to prevent overfitting at closer ranges

    rr = np.zeros((n,prev_num))
    r = np.zeros((n,prev_num))
    l = np.zeros((n,prev_num))
    ll = np.zeros((n,prev_num))
    rrstd = np.zeros((n,prev_num))
    rstd = np.zeros((n,prev_num))
    lstd = np.zeros((n,prev_num))
    llstd = np.zeros((n,prev_num))
    le = np.zeros((n,prev_num))
    re = np.zeros((n,prev_num))
    lestd = np.zeros((n,prev_num))
    restd = np.zeros((n,prev_num))

    rr_inter = np.zeros((n,prev_num2))
    r_inter = np.zeros((n,prev_num2))
    l_inter = np.zeros((n,prev_num2))
    ll_inter = np.zeros((n,prev_num2))
    rrstd_inter = np.zeros((n,prev_num2))
    rstd_inter = np.zeros((n,prev_num2))
    lstd_inter = np.zeros((n,prev_num2))
    llstd_inter = np.zeros((n,prev_num2))
    le_inter = np.zeros((n,prev_num2))
    re_inter = np.zeros((n,prev_num2))
    lestd_inter = np.zeros((n,prev_num2))
    restd_inter = np.zeros((n,prev_num2))

    rr_prob = np.zeros((n,1))
    r_prob = np.zeros((n,1))
    l_prob = np.zeros((n,1))
    ll_prob = np.zeros((n,1))

    t_ = tb.capnp_df["modelV2"]["Timestamp"].to_numpy()
    ll_ = tb.capnp_df["modelV2"]["modelV2.laneLines.0.y"].to_numpy()
    l_ = tb.capnp_df["modelV2"]["modelV2.laneLines.1.y"].to_numpy()
    r_ = tb.capnp_df["modelV2"]["modelV2.laneLines.2.y"].to_numpy()
    rr_ = tb.capnp_df["modelV2"]["modelV2.laneLines.3.y"].to_numpy()
    le_ = tb.capnp_df["modelV2"]["modelV2.roadEdges.0.y"].to_numpy()
    re_ = tb.capnp_df["modelV2"]["modelV2.roadEdges.1.y"].to_numpy()

    llstd_ = tb.capnp_df["modelV2"]["modelV2.laneLines.0.yStd"].to_numpy()
    lstd_ = tb.capnp_df["modelV2"]["modelV2.laneLines.1.yStd"].to_numpy()
    rstd_ = tb.capnp_df["modelV2"]["modelV2.laneLines.2.yStd"].to_numpy()
    rrstd_ = tb.capnp_df["modelV2"]["modelV2.laneLines.3.yStd"].to_numpy()
    lestd_ = tb.capnp_df["modelV2"]["modelV2.roadEdges.0.yStd"].to_numpy()
    restd_ = tb.capnp_df["modelV2"]["modelV2.roadEdges.1.yStd"].to_numpy()


    lane_prob = tb.capnp_df["modelV2"]["modelV2.laneLineProbs"].to_numpy()

    for i in range(n):
        for j in range(prev_num):
            ll[i,j] = ll_[i][j]
            l[i,j] = l_[i][j]
            r[i,j] = r_[i][j]
            rr[i,j] = rr_[i][j]
            le[i,j] = le_[i][j]
            re[i,j] = re_[i][j]
            
            llstd[i,j] = llstd_[i][j]
            lstd[i,j] = lstd_[i][j]
            rstd[i,j] = rstd_[i][j]
            rrstd[i,j] = rrstd_[i][j]
            
            lestd[i,j] = lestd_[i][j]
            restd[i,j] = restd_[i][j]
        
        # Interpolation according to x_:[0 10 20 ... 100]
        ll_inter[i,:] = np.interp(x_,Data['lane']['x'],ll[i,:])
        l_inter[i,:] = np.interp(x_,Data['lane']['x'],l[i,:])
        r_inter[i,:] = np.interp(x_,Data['lane']['x'],r[i,:])
        rr_inter[i,:] = np.interp(x_,Data['lane']['x'],rr[i,:])
        llstd_inter[i,:] = np.interp(x_,Data['lane']['x'],llstd[i,:])
        lstd_inter[i,:] = np.interp(x_,Data['lane']['x'],lstd[i,:])
        rstd_inter[i,:] = np.interp(x_,Data['lane']['x'],rstd[i,:])
        rrstd_inter[i,:] = np.interp(x_,Data['lane']['x'],rrstd[i,:])
        le_inter[i,:] = np.interp(x_,Data['lane']['x'],le[i,:])
        re_inter[i,:] = np.interp(x_,Data['lane']['x'],re[i,:])
        lestd_inter[i,:] = np.interp(x_,Data['lane']['x'],lestd[i,:])
        restd_inter[i,:] = np.interp(x_,Data['lane']['x'],restd[i,:])
        
        
        ll_prob[i] = lane_prob[i][0]
        l_prob[i] = lane_prob[i][1]
        r_prob[i] = lane_prob[i][2]
        rr_prob[i] = lane_prob[i][3]

    Data['lane']['rr'] = rr
    Data['lane']['r'] = r
    Data['lane']['l'] = l
    Data['lane']['ll'] = ll
    Data['lane']['rrstd'] = rrstd
    Data['lane']['rstd'] = rstd
    Data['lane']['lstd'] = lstd
    Data['lane']['llstd'] = llstd
    Data['lane']['le'] = le
    Data['lane']['re'] = re
    Data['lane']['lestd'] = lestd
    Data['lane']['restd'] = restd
    Data['lane']['t'] = t_

    Data['lane']['ll_inter'] = ll_inter
    Data['lane']['l_inter'] = l_inter
    Data['lane']['r_inter'] = r_inter
    Data['lane']['rr_inter'] = rr_inter
    Data['lane']['llstd_inter'] = llstd_inter
    Data['lane']['lstd_inter'] = lstd_inter
    Data['lane']['rstd_inter'] = rstd_inter
    Data['lane']['rrstd_inter'] = rrstd_inter
    Data['lane']['le_inter'] = le_inter
    Data['lane']['re_inter'] = re_inter
    Data['lane']['lestd_inter'] = lestd_inter
    Data['lane']['restd_inter'] = restd_inter

    Data['lane']['ll_prob'] = ll_prob
    Data['lane']['l_prob'] = l_prob
    Data['lane']['r_prob'] = r_prob
    Data['lane']['rr_prob'] = rr_prob

    # CAN Data
    # Data['can'] = {}

    # Data['can']['ws_rl'] = tb.capnp_df["carState"]["carState.wheelSpeeds.rl"].to_numpy()
    # Data['can']['ws_rr'] = tb.capnp_df["carState"]["carState.wheelSpeeds.rl"].to_numpy()
    # Data['can']['leftBlinker'] = tb.capnp_df["carState"]["carState.leftBlinker"].to_numpy()
    # Data['can']['rightBlinker'] = tb.capnp_df["carState"]["carState.rightBlinker"].to_numpy()                  
    # Data['can']['t'] = tb.capnp_df["carState"]["Timestamp"].to_numpy()                                

    # GNSS
    Data['gnss'] = {}
    Data['gnss']['lat'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.latitude"].to_numpy()
    Data['gnss']['lon'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.longitude"].to_numpy()                                  
    Data['gnss']['alt'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.altitude"].to_numpy()
    Data['gnss']['hAcc'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.accuracy"].to_numpy()    
    Data['gnss']['vAcc'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.verticalAccuracy"].to_numpy()
    Data['gnss']['bearingDeg'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.bearingDeg"].to_numpy()
    Data['gnss']['bearingDegAcc'] = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.bearingAccuracyDeg"].to_numpy()
    Data['gnss']['valid'] = tb.capnp_df["gpsLocationExternal"]["valid"].to_numpy()      

    Data['gnss']['t'] = tb.capnp_df["gpsLocationExternal"]["Timestamp"].to_numpy() 

    n = tb.capnp_df["gpsLocationExternal"]["Timestamp"].to_numpy().size

    vNED_ = tb.capnp_df["gpsLocationExternal"]["gpsLocationExternal.vNED"].to_numpy() 

    vNED = np.zeros((n,3))
    for i in range(n):
        vNED[i,0] = vNED_[i][0]
        vNED[i,1] = vNED_[i][1]
        vNED[i,2] = vNED_[i][2]

    Data['gnss']['vNED'] = vNED

    ## IMU
    Data['imu'] = {}

    t = tb.capnp_df["sensorEvents"][["Timestamp","sensorEvents"]].to_numpy()

    cnt_accel = 0
    cnt_gyro = 0
    cnt_calibgyro = 0

    for i in range(len(t)):
        for SensorInfo in t[i,1]:
            if 'acceleration' in SensorInfo:
                cnt_accel += 1
            elif 'gyroUncalibrated' in SensorInfo:
                cnt_gyro += 1

    accel_dict = {}
    ax_ = np.zeros((cnt_accel,1))
    ay_ = np.zeros((cnt_accel,1))
    az_ = np.zeros((cnt_accel,1))
    acct_ = np.zeros((cnt_accel,1))

    gyro_dict = {}
    wx_ = np.zeros((cnt_gyro,1))
    wy_ = np.zeros((cnt_gyro,1))
    wz_ = np.zeros((cnt_gyro,1))
    gyrot_ = np.zeros((cnt_gyro,1))


    cnt_accel = 0
    cnt_gyro = 0

    for i in range(len(t)):
        for SensorInfo in t[i,1]:            
            if 'acceleration' in SensorInfo:  
                acct_[cnt_accel] = t[i,0]
                ax_[cnt_accel] = np.array(SensorInfo['acceleration']['v'][0])
                ay_[cnt_accel] = np.array(SensorInfo['acceleration']['v'][1])
                az_[cnt_accel] = np.array(SensorInfo['acceleration']['v'][2])
                cnt_accel += 1
            elif 'gyroUncalibrated' in SensorInfo:
                gyrot_[cnt_gyro] = t[i,0]
                wx_[cnt_gyro] = np.array(SensorInfo['gyroUncalibrated']['v'][0])
                wy_[cnt_gyro] = np.array(SensorInfo['gyroUncalibrated']['v'][1])
                wz_[cnt_gyro] = np.array(SensorInfo['gyroUncalibrated']['v'][2])
                cnt_gyro += 1

    Data['imu']['a_t'] = acct_
    Data['imu']['ax'] = ax_
    Data['imu']['ay'] = ay_
    Data['imu']['az'] = az_

    Data['imu']['w_t'] = gyrot_
    Data['imu']['wx'] = wx_
    Data['imu']['wy'] = wy_
    Data['imu']['wz'] = wz_

    return Data 

def Extension(Data,ExtData):
    """
    Concatenate separated dictionaries of same keys    
    
    If input is 'None', initialized dictionary is returned
    Else, reference dictionary 'Data' is extended by 'ExtData' and returned
    """
    # Initialization
    if Data is None and ExtData is None:
        Data = {} # Full Data dictionary
        Data['imu'] = {}
        Data['imu']['a_t'] = []
        Data['imu']['ax'] = []
        Data['imu']['ay'] = []
        Data['imu']['az'] = []
        Data['imu']['w_t'] = []
        Data['imu']['wx'] = []
        Data['imu']['wy'] = []
        Data['imu']['wz'] = []

        Data['gnss'] = {}
        Data['gnss']['t'] = []
        Data['gnss']['lat'] = []
        Data['gnss']['lon'] = []
        Data['gnss']['alt'] = []
        Data['gnss']['hAcc'] = []
        Data['gnss']['vAcc'] = []
        Data['gnss']['bearingDeg'] = []
        Data['gnss']['bearingDegAcc'] = []
        Data['gnss']['valid'] = []
        Data['gnss']['vNED'] = []

        Data['lane'] = {}
        Data['lane']['rr'] = []
        Data['lane']['r'] = []
        Data['lane']['l'] = []
        Data['lane']['ll'] = []
        Data['lane']['rrstd'] = []
        Data['lane']['rstd'] = []
        Data['lane']['lstd'] = []
        Data['lane']['llstd'] = []
        Data['lane']['le'] = []
        Data['lane']['re'] = []
        Data['lane']['lestd'] = []
        Data['lane']['restd'] = []
        Data['lane']['t'] = []

        Data['lane']['ll_inter'] = []
        Data['lane']['l_inter'] = []
        Data['lane']['r_inter'] = []
        Data['lane']['rr_inter'] = []
        Data['lane']['llstd_inter'] = []
        Data['lane']['lstd_inter'] = []
        Data['lane']['rstd_inter'] = []
        Data['lane']['rrstd_inter'] = []
        Data['lane']['le_inter'] = []
        Data['lane']['re_inter'] = []
        Data['lane']['lestd_inter'] = []
        Data['lane']['restd_inter'] = []

        Data['lane']['ll_prob'] = []
        Data['lane']['l_prob'] = []
        Data['lane']['r_prob'] = []
        Data['lane']['rr_prob'] = []
    
    else:
        Data['imu']['a_t'].extend(ExtData['imu']['a_t'])
        Data['imu']['ax'].extend(ExtData['imu']['ax'])
        Data['imu']['ay'].extend(ExtData['imu']['ay'])
        Data['imu']['az'].extend(ExtData['imu']['az'])
        Data['imu']['w_t'].extend(ExtData['imu']['w_t'])
        Data['imu']['wx'].extend(ExtData['imu']['wx'])
        Data['imu']['wy'].extend(ExtData['imu']['wy'])
        Data['imu']['wz'].extend(ExtData['imu']['wz'])

        Data['gnss']['t'].extend(ExtData['gnss']['t'])
        Data['gnss']['lat'].extend(ExtData['gnss']['lat'])
        Data['gnss']['lon'].extend(ExtData['gnss']['lon'])
        Data['gnss']['alt'].extend(ExtData['gnss']['alt'])
        Data['gnss']['hAcc'].extend(ExtData['gnss']['hAcc'])
        Data['gnss']['vAcc'].extend(ExtData['gnss']['vAcc'])
        Data['gnss']['bearingDeg'].extend(ExtData['gnss']['bearingDeg'])
        Data['gnss']['bearingDegAcc'].extend(ExtData['gnss']['bearingDegAcc'])
        Data['gnss']['valid'].extend(ExtData['gnss']['valid'])
        Data['gnss']['vNED'].extend(ExtData['gnss']['vNED'])

        Data['lane']['rr'].extend(ExtData['lane']['rr'])
        Data['lane']['r'].extend(ExtData['lane']['r'])
        Data['lane']['l'].extend(ExtData['lane']['l'])
        Data['lane']['ll'].extend(ExtData['lane']['ll'])
        Data['lane']['rrstd'].extend(ExtData['lane']['rrstd'])
        Data['lane']['rstd'].extend(ExtData['lane']['rstd'])
        Data['lane']['lstd'].extend(ExtData['lane']['lstd'])
        Data['lane']['llstd'].extend(ExtData['lane']['llstd'])
        Data['lane']['le'].extend(ExtData['lane']['le'])
        Data['lane']['re'].extend(ExtData['lane']['re'])
        Data['lane']['lestd'].extend(ExtData['lane']['lestd'])
        Data['lane']['restd'].extend(ExtData['lane']['restd'])
        Data['lane']['t'].extend(ExtData['lane']['t'])

        Data['lane']['ll_inter'].extend(ExtData['lane']['ll_inter'])
        Data['lane']['l_inter'].extend(ExtData['lane']['l_inter'])
        Data['lane']['r_inter'].extend(ExtData['lane']['r_inter'])
        Data['lane']['rr_inter'].extend(ExtData['lane']['rr_inter'])
        Data['lane']['llstd_inter'].extend(ExtData['lane']['llstd_inter'])
        Data['lane']['lstd_inter'].extend(ExtData['lane']['lstd_inter'])
        Data['lane']['rstd_inter'].extend(ExtData['lane']['rstd_inter'])
        Data['lane']['rrstd_inter'].extend(ExtData['lane']['rrstd_inter'])
        Data['lane']['le_inter'].extend(ExtData['lane']['le_inter'])
        Data['lane']['re_inter'].extend(ExtData['lane']['re_inter'])
        Data['lane']['lestd_inter'].extend(ExtData['lane']['lestd_inter']) 
        Data['lane']['restd_inter'].extend(ExtData['lane']['restd_inter'])

        Data['lane']['ll_prob'].extend(ExtData['lane']['ll_prob'])
        Data['lane']['l_prob'].extend(ExtData['lane']['l_prob'])
        Data['lane']['r_prob'].extend(ExtData['lane']['r_prob'])
        Data['lane']['rr_prob'].extend(ExtData['lane']['rr_prob'])

    return Data


if __name__ == '__main__':
    # Target folder paths
    folder_paths = ["D/SJ_Dataset/2023/2023-01-26--18-39-41/","D/SJ_Dataset/2023/2023-01-26--19-49-58/"]

    for folder_path in folder_paths:
        print("=====================================================================")
        print("Data Processing for Folder " + folder_path)
        InputLogs = glob(folder_path + "*" + "/rlog.bz2")
        L = len(InputLogs)
        Data = Extension(None,None)

        # For large Data, divide into several fragments and process
        if L > 10:
            r = L % 10
            p = L // 10
            for i in range(p):
                if i > 0:
                    search_strings = str(i) + '[0-9]'
                else:
                    search_strings = '[0-9]'

                SubInputLogs = glob(folder_path + search_strings + "/rlog.bz2")
                print("Data Processing rlog folders " + str(i) + "0 ~ " + str(i) + "9")

                tb = Tabulator(MultiLogIterator(SubInputLogs), None)
                tb.capnp_to_pandas()
                tb._capnp_unixtime()

                data = Processor(tb)
                Data = Extension(Data,data)

            # Processing last folders
            search_strings = str(p) + '[0-' + str(r-1) + ']'
            SubInputLogs = glob(folder_path + search_strings + "/rlog.bz2")
            print("Data Processing rlog folders " + str(p) + "0 ~ " + str(p) + str(r-1))
            tb = Tabulator(MultiLogIterator(SubInputLogs), None)
            tb.capnp_to_pandas()
            tb._capnp_unixtime()

            data = Processor(tb)
            Data = Extension(Data,data)
            print("Saving augmented data into .mat format...")
            sio.savemat(folder_path + 'Data.mat',Data,oned_as='row')
    print("=====================================================================")
    print("Data Processing for requested folders finished")

