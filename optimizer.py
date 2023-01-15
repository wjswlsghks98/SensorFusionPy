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

    def optimize(self):
        print("Optimization Starts...")

        n = len(self.states)

        if (self.mode != '2-phase') and (self.mode != 'full'):
            # Run optimization depending on algorithm options
            #
            # [Basic Mode]: INS + GNSS Fusion
            # [Partial Mode]: INS + GNSS + WSS Fusion
            
            if self.mode == 'basic':
                x0 = np.zeros((15*n,1))
            elif self.mode == 'partial':
                x0 = np.zeros((16*n,1))

            if self.options.Algorithm == 'GN':
                self.GaussNewton(x0)
            elif self.options.Algorithm == 'TR':
                self.TrustRegion(x0)

    def GaussNewton(self,x0):
        """
        Gauss-Newton method shows very fast convergence compared to
        Trust-Region method, but suffers from low convergence stability
        near the local minima (oscillation frequently observed)
        Oscillation detection was implemented to reduce meaningless iterations

        """
        print("[SNLS Solver: Gauss-Newton Method]")
        print("Iteration            f(x)              step size")

        res, jac = self.cost_func(x0)
        prev_cost = np.linalg.norm(res)**2
        print('  {0:3d}   {1:7d}   {2:5d}'.format(0,prev_cost,np.linalg.norm(x0)))

        i=1
        cost_stack = [prev_cost]

        while True:
            A = jac.T @ jac # Check if @ is supported for sparse matrix multiplication
            b = -jac.T @ res

            x0 = sp.sparse.linalg.spsolve(A,b)
            res, jac = self.cost_func(x0)
            cost = np.linalg.norm(res)**2
            cost_stack.append(cost)
            step_size = np.linalg.norm(x0)

            print('  {0:3d}   {1:7d}   {2:5d}'.format(i,cost,step_size))

            # Ending Criterion
            flags = []
            flags.append(np.abs(prev_cost - cost) > self.options.CostThres)
            flags.append(step_size > self.options.StepThres)
            flags.append(i < self.options.IterThres)

            # Check for oscillation around the local minima
            if len(cost_stack) >= 5:
                osc_flag = self.detectOsc(cost_stack)
            else:
                osc_flag = False

            if np.sum(np.array(flags)) != len(flags):
                self.retract(x0,'final')
                print("Optimization Finished...")
                idx = np.where(flags == 0)[0]

                if idx == 0:
                    print('Current Cost Difference {0} is below threshold {1}'.format(np.abs(prev_cost - cost), self.options.CostThres))
                elif idx == 1:
                    print('Current Step Size {0} is below threshold {1}'.format(step_size, self.options.StepThres))
                elif idx == 2:
                    print('Current Iteration Number {0} is above threshold {1}'.format(i, self.options.IterThres))
                
                break
            elif osc_flag:
                self.retract(x0,'final')
                print("Optimization Finished...")
                print("Oscillation about the local minima detected")
                break
            else:
                i += 1
                prev_cost = cost

    def TrustRegion(self,x0):
        """
        Indefinite Gauss-Newton-Powell's Dog-Leg algorithm
        Implemented "RISE: An Incremental Trust Region Method for Robust Online Sparse Least-Squares Estimation"
        by David M. Rosen etal
        
        Implemented in Python3 by JinHwan Jeon, 2023
        
        Original paper considers case for rank-deficient Jacobians,
        but in this sensor fusion framework, Jacobian must always be
        full rank. Therefore, 'rank-deficient Jacobian' part of the
        algorithm is not implemented.
        
        Moreover, the core part of the algorithm is "incremental"
        trust region method, but here the optimization is done in a
        batch manner. Therefore incremental solver framework is not
        adopted in this implementation.
        
        """

    def cost_func(self,x0):
        self.retract(x0,'normal')
        Pr_res, Pr_jac = self.CreatePrBlock()
        MM_res, MM_jac = self.CreateMMBlock()
        GNSS_res, GNSS_jac = self.CreateGNSSBlock()
        WSS_res, WSS_jac = self.CreateWSSBlock()

        res = sp.sparse.vstack([Pr_res, MM_res, GNSS_res, WSS_res])
        jac = sp.sparse.vstack([Pr_jac, MM_jac, GNSS_jac, WSS_jac])

        return res, jac

    def CreatePrBlock(self):
        """
        
        """

    def CreateMMBlock(self):
        """
        
        """

    def CreateGNSSBlock(self):
        """
        
        """

    def CreateWSSBlock(self):
        """
        
        """
    
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
            deltaR = vert(state_delta[9*(i-1):9*(i-1)+3])
            deltaV = vert(state_delta[9*(i-1)+3:9*(i-1)+6])
            deltaP = vert(state_delta[9*(i-1)+6:9*(i-1)+9])

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
            bgd_ = vert(bias_ddelta[6*(i-1):6*(i-1)+3])
            bad_ = vert(bias_ddelta[6*(i-1)+3:6*(i-1)+6])
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
            
            elif mode == 'normal':
                # If delta bias values are larger than threshold,
                # perform repropgation (IMU)
                # Else, accumulate delta values

                if np.linalg.norm(new_bgd) > self.bgd_thres:
                    bg_ = self.bias[i].bg + new_bgd
                    bgd_ = np.zeros((3,1))
                    flag = True
                else:
                    bg_ = self.bias[i].bg
                    bgd_ = new_bgd
                
                if np.linalg.norm(new_bad) > self.bad_thres:
                    ba_ = self.bias[i].ba + new_bad
                    bad_ = np.zeros((3,1))
                    flag = True
                else:
                    ba_ = self.bias[i].ba
                    bad_ = new_bad
            
            if flag and i != n-1:
                idxs.append(i)
            
            self.bias[i].bg = bg_
            self.bias[i].ba = ba_
            self.bias[i].bgd = bgd_
            self.bias[i].bad = bad_

        if len(idxs) > 0:
            self.integrate(idxs)

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

    @staticmethod
    def detectOsc(cost_stack):
        last_five = np.array(cost_stack[-5::1])
        avg = np.mean(last_five)
        delta = last_five - avg
        if np.max(delta) < 1e2:
            return True
        else:
            return False

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
    
    