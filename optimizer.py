import numpy as np
import scipy as sp
from operators import *
from math import *
import pymap3d as pm

class Optimizer:
    """
    Sparse Nonlinear Least Squares based Multi-modal Sensor Fusion Framework
    
    [Mode]
    * 'Basic': INS + GNSS Fusion
    * 'Partial': INS + GNSS + WSS Fusion
    * '2-phase': INS + GNSS + WSS + Lane Fusion --> Currently inaccurate

    Implemented by JinHwan Jeon, 2023
    """
    def __init__(self,**kwargs):
        self.mode = kwargs['mode'] # Optimization mode
        self.imu = kwargs['imu'] # IMU: list format 
        self.gnss = kwargs['gnss'] # GNSS
        self.lane = kwargs['lane'] # Lane
        self.snap = kwargs['snap'] # Road-to-Snap
        self.imu_bias = kwargs['imu_bias'] # Initial guess of imu bias
        self.t = kwargs['t'] # Time
        self.covs = kwargs['covs'] # Covariance matrices for sensors
        self.options = kwargs['options'] # Optimization options
        self.bias = [] # Bias variables saver
        self.states = [] # State variables saver
        self.bgd_thres = 1e-2 # Gyro bias repropagation threshold
        self.bad_thres = 1e-3 # Accel bias repropagation threshold
        self.params = vert(np.array([1.5,0,1]))
        
        for i in range(len(self.imu)):
            bias = Bias(self.imu_bias)
            self.bias.append(bias)
        
        # For initialization, integrate IMU over full timestep
        self.integrate(list(range(len(self.imu))))

    def integrate(self,idxs):
        """
        IMU Pre-integration based on the paper of C.Forster etal, 2015
        
        Perform repropagation if bias updates are larger than the repropagation threshold
        """
        nbg_cov = self.covs.imu.GyroscopeNoise # Should be already in np array format
        nba_cov = self.covs.imu.AccelerometerNoise # Should be already in np array format
        n_cov = sp.linalg.block_diag(nbg_cov,nba_cov)

        for idx in idxs:

            # self.imu should be defined in data processing process
            # Need imu class
            n =len(self.imu[idx].accel) # IMU accel data in row separated form [ax, ay, az]
            t = self.imu[idx].t
            a = (self.imu[idx].accel).T
            w = (self.imu[idx].gyro).T
            ab = vert(self.bias[idx].ba)
            wb = vert(self.bias[idx].bg)

            delRik = np.eye(3); delVik = np.zeros((3,1)); delPik = np.zeros((3,1))
            JdelRik_bg = np.zeros((3,3))
            JdelVik_bg = np.zeros((3,3))
            JdelVik_ba = np.zeros((3,3))
            JdelPik_bg = np.zeros((3,3))
            JdelPik_ba = np.zeros((3,3))

            noise_cov = np.zeros((9,9))
            noise_vec = np.zeros((9,1))

            dtij = 0

            for k in range(n):
                dt_k = t[k+1] - t[k]
                dtij += dt_k

                a_k = vert(a[:,k]) - ab
                w_k = vert(w[:,k]) - wb

                delRkkp1 = Exp_map(w_k * dt_k)
                Ak, Bk = self.getCoeff(delRik,delRkkp1,a_k,w_k,dt_k)                
                noise_cov = Ak @ noise_cov @ Ak.T + Bk @ (1/dt_k * n_cov) @ Bk.T

                # IMU Measurement Propagation
                delPik = delPik + delVik * dt_k + 1/2 * delRik @ a_k * dt_k**2
                delVik = delVik + delRik @ a_k * dt_k

                # Jacobian Propagation
                JdelPik_ba = JdelPik_ba + JdelVik_ba * dt_k - 1/2 * delRik * dt_k**2
                JdelPik_bg = JdelPik_bg + JdelVik_bg * dt_k - 1/2 * delRik @ skew(a_k) @ JdelRik_bg * dt_k**2
                JdelVik_ba = JdelVik_ba - delRik * dt_k
                JdelVik_bg = JdelVik_bg - delRik @ skew(a_k) @ JdelRik_bg * dt_k
                JdelRik_bg = delRkkp1.T @ JdelRik_bg - RightJac(w_k * dt_k) * dt_k

                delRik = delRik * Exp_map(w_k * dt_k)

            self.imu[idx].JdelRij_bg = JdelRik_bg
            self.imu[idx].JdelVij_bg = JdelVik_bg
            self.imu[idx].JdelVij_ba = JdelVik_ba
            self.imu[idx].JdelPij_bg = JdelPik_bg
            self.imu[idx].JdelPij_ba = JdelPik_ba
            self.imu[idx].delRij = delRik
            self.imu[idx].delVij = delVik
            self.imu[idx].delPij = delPik
            self.imu[idx].Covij = noise_cov
            self.imu[idx].nij = noise_vec

            w, v = np.linalg.eig(noise_cov)
            m = np.sum(w >= 0)
            if m != len(w):
                # There exists non-positive eigenvalue for covariance matrix
                raise ValueError('Computed IMU Covariance is not positive definite')

            self.imu[idx].dtij = dtij

    def ins(self):
        """
        INS Propagation in the world frame using pre-integrated values

        Vehicle Body Frame: [x, y, z] = [Forward, Left, Up]
        World Frame: [x, y, z] = [East, North, Up]
        Camera Frame(Phone): https://source.android.com/docs/core/sensors/sensor-types
        
        Note that IMU (Android) measurements have gravitational effects, 
        which should be considered when modeling vehicle 3D kinematics

        Vehicle States Variables
        - R: Body-to-world frame rotational matrix
        - V: World frame velocity vector
        - P: World frame position vector (local frame coords, not geodetic)

        Implemented by JinHwan Jeon, 2023
        
        """
        print("INS Propagation")
        th = pi/2 - pi/180 * self.gnss.bearing[0]
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0],
                      [0, 0, 1]])
        Vned = self.gnss.vNED[0,:]
        V = vert(np.array([Vned[1],Vned[0],-Vned[2]]))
        lat, lon, alt = pm.geodetic2enu(self.gnss.pos[0,0],self.gnss.pos[0,1],self.gnss.pos[0,2],
                            self.gnss.lla0[0],self.gnss.lla0[1],self.gnss.lla0[2])
        P = vert(np.array([lat, lon, alt]))
        grav = vert(np.array([0,0,-9.81]))
        
        state = State(R,V,P)

        # Add Lane information in the future

        self.states.append(state)

        # For initial state(only), add bias information for convenience
        state.bg = self.bias[0].bg
        state.ba = self.bias[0].ba
        state.bgd = self.bias[0].bgd
        state.bad = self.bias[0].bad
        state.L = self.params

        self.init_state = state

        for imu in self.imu:
            delRij = imu.delRij
            delVij = imu.delVij
            delPij = imu.delPij
            dtij = imu.dtij

            P = P + V * dtij + 1/2 * grav * dtij**2 + R @ delPij
            V = V + grav * dtij + R @ delVij
            R = R * delRij
            state = State(R,V,P)
            self.states.append(state)

    def cost_func(self,x0):
        """
        Augment cost function residual and jacobian

        """
        self.retract(x0,'normal')
        

    
    def retract(self,delta,mode):
        """
        Update variables using delta values

        """
        n = len(self.states)
        state_delta = delta[0:9*n]
        bias_ddelta = delta[9*n:15*n]

        self.retractStates(state_delta)
        self.retractBias(bias_ddelta,mode)

        if self.mode != 'basic':
            wsf_delta = delta[15*n:16*n]
            self.retractWSF(wsf_delta)


    def retractStates(self,state_delta):
        n = len(self.states)
        for i in range(n):
            deltaR = state_delta[9*(i-1):9*(i-1)+3]
            deltaV = state_delta[9*(i-1)+3:9*(i-1)+6]
            deltaP = state_delta[9*(i-1)+6:9*(i-1)+9]

            R = self.states[i].R
            V = self.states[i].V
            P = self.states[i].P

            P_ = P + R @ deltaP
            V_ = V + deltaV
            R_ = R @ Exp_map(deltaR)

            self.states[i].R = R_
            self.states[i].V = V_
            self.states[i].P = P_

    def retractBias(self,bias_ddelta,mode):
        n = len(self.bias)
        idxs = []
        for i in range(n):
            bgd_ = bias_ddelta[6*(i-1):6*(i-1)+3]
            bad_ = bias_ddelta[6*(i-1)+3:6*(i-1)+6]
            new_bgd = bgd_ + self.bias[i].bgd
            new_bad = bad_ + self.bias[i].bad

            flag = False

            if mode == 'final':
                # After finishing optimization, all delta terms are 
                # transferred to the original bias variables
                # Flush delta values!
                bg_ = self.bias[i].bg + new_bgd
                bgd_ = np.zeros((3,1))
                ba_ = self.bias[i].ba + new_bad
                bad_ = np.zeros((3,1))

    def retractWSF(self,wsf_delta):
        """
        
        """


    @staticmethod
    def getCoeff(delRik,delRkkp1,a_k,w_k,dt_k):
        Ak = np.zeros((9,9))
        Bk = np.zeros((9,6))

        Ak[0:3,0:3] = delRkkp1.T
        Ak[3:6,0:3] = -delRik @ skew(a_k) * dt_k
        Ak[3:6,3:6] = np.eye(3)
        Ak[6:9,0:3] = -1/2 * delRik @ skew(a_k) * dt_k**2
        Ak[6:9,3:6] = np.eye(3) * dt_k
        Ak[6:9,6:9] = np.eye(3)

        Bk[0:3,0:3] = RightJac(w_k * dt_k) * dt_k
        Bk[3:6,3:6] = delRik * dt_k
        Bk[6:9,3:6] = 1/2 * delRik * dt_k**2

        return Ak, Bk

class Bias:
    def __init__(self,imu_bias):
        self.bg = imu_bias['gyro']
        self.ba = imu_bias['accel']
        self.bgd = np.zeros((3,1))
        self.bad = np.zeros((3,1))       

class State:
    """
    Definition of 3D Vehicle States 
    
    """
    def __init__(self,R,V,P):
        self.R = R # 3D Rotational Matrix(Attitude)
        self.V = V # Velocity Vector
        self.P = P # Position Vector
        self.WSF = 1 # Wheel Speed Scaling Factor

if __name__ == '__main__':
    # imu_bias = dict(gyro=np.array([0,0,0]),accel=np.array([0,0,0]))
    # sample = dict(mode=1,imu=[0,0,0,0],gnss=1,lane=1,snap=1,imu_bias=imu_bias,t=1,covs=1,options=1)
    # testing = Optimizer(**sample)
    # print(testing.bias[0].ba)
    a = np.array([[1,2,3],[4,5,6]])
    
    