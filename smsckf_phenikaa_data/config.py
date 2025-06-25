import numpy as np
import cv2


class OptimizationConfigEuRoC(object):
    """
    Configuration parameters for 3d feature position optimization.
    """
    def __init__(self):
        self.translation_threshold = 0.2
        self.huber_epsilon = 0.01
        self.estimation_precision = 5e-7
        self.initial_damping = 1e-3
        self.outer_loop_max_iteration = 10
        self.inner_loop_max_iteration = 10

class ConfigEuRoC(object):
    def __init__(self):
        # feature position optimization
        self.optimization_config = OptimizationConfigEuRoC()

        ## image processor
        self.grid_row = 7
        self.grid_col = 8
        self.grid_num = self.grid_row * self.grid_col
        self.grid_min_feature_num = 50
        self.grid_max_feature_num = 100
        self.fast_threshold = 10
        self.ransac_threshold = 3.0
        self.stereo_threshold = 5.0
        self.max_iteration = 30
        self.track_precision = 0.01
        self.pyramid_levels = 0
        self.patch_size = 51
        self.win_size = (self.patch_size, self.patch_size)

        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                self.max_iteration, 
                self.track_precision),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        

        ## msckf vio
        # gravity
        self.gravity_acc = 9.81
        self.gravity = np.array([0.0, 0.0, -self.gravity_acc])

        # Framte rate of the stereo images. This variable is only used to 
        # determine the timing threshold of each iteration of the filter.
        self.frame_rate = 30

        # Maximum number of camera states to be stored
        self.max_cam_state_size = 20

        # The position uncertainty threshold is used to determine
        # when to reset the system online. Otherwise, the ever-increaseing
        # uncertainty will make the estimation unstable.
        # Note this online reset will be some dead-reckoning.
        # Set this threshold to nonpositive to disable online reset.
        self.position_std_threshold = 8.0

        # Threshold for determine keyframes
        self.rotation_threshold = 0.2
        self.translation_threshold = 0.3
        self.tracking_rate_threshold = 0.3

        # Noise related parameters (Use variance instead of standard deviation)

        self.gyro_noise = (1.9606e-5) ** 2
        self.acc_noise = 0.0539 ** 2
        self.gyro_bias_noise = (2.1142e-7) ** 2
        self.acc_bias_noise = 0.0006 ** 2
        self.observation_noise = 0.001 ** 2

        # initial state
        self.velocity = np.zeros(3)

        # The initial covariance of orientation and position can be
        # set to 0. But for velocity, bias and extrinsic parameters, 
        # there should be nontrivial uncertainty.

        self.velocity_cov = 0.009
        self.gyro_bias_cov = 1.9606e-5
        self.acc_bias_cov = 0.0539
        self.extrinsic_rotation_cov = 3.0462e-4
        self.extrinsic_translation_cov = 2.5e-5


        ## calibration parameters
        # T_imu_cam: takes a vector from the IMU frame to the cam frame.
        # T_cn_cnm1: takes a vector from the cam0 frame to the cam1 frame.
        # see https://github.com/ethz-asl/kalibr/wiki/yaml-formats
        
        self.T_imu_cam0 = np.array([
            [ 1.0000,   0.0000,   0.0000,   1.5090],
            [ 0.0000,   1.0000,   0.0000,   0.0000],
            [ 0.0000,   0.0000,   1.0000,   1.3500],
            [ 0.0000,   0.0000,   0.0000,   1.0000]]) # imu frame -> mf frame

        self.cam0_camera_model = 'pinhole'
        self.cam0_distortion_model = 'radtan'
        self.cam0_distortion_coeffs = np.array(
            [-0.536735, -0.488961, -0.000233, 0.001435, -1.027051])
        self.cam0_intrinsics = np.array([957.865, 981.29, 480, 270]) # mf
        self.cam0_resolution = np.array([960, 540])


        self.T_imu_cam1 = np.array([
            [ 1.0000,   0.0000,   0.0000,   1.5090],
            [ 0.0000,   1.0000,   0.0000,  -0.1000],
            [ 0.0000,   0.0000,   1.0000,   1.6500],
            [ 0.0000,   0.0000,   0.0000,   1.0000]]) # imu frame -> wf frame

        # self.T_imu_cam1 = np.array([
        #     [ 1.0000,   0.0000,   0.0000,   1.5090],
        #     [ 0.0000,   1.0000,   0.0000,   0.1000],
        #     [ 0.0000,   0.0000,   1.0000,   1.3500],
        #     [ 0.0000,   0.0000,   0.0000,   1.0000]]) # imu frame -> nf frame


        self.T_cn_cnm1 = np.array([
            [ 1.0000,   0.0000,   0.0000,   0.0000],
            [ 0.0000,   1.0000,   0.0000,  -0.1000],
            [ 0.0000,   0.0000,   1.0000,   0.3000],
            [ 0.0000,   0.0000,   0.0000,   1.0000]]) # mf frame -> wf frame

        # self.T_cn_cnm1 = np.array([
        #     [ 1.0000,   0.0000,   0.0000,   0.0000],
        #     [ 0.0000,   1.0000,   0.0000,   0.1000],
        #     [ 0.0000,   0.0000,   1.0000,   0.0000],
        #     [ 0.0000,   0.0000,   0.0000,   1.0000]]) # mf frame -> nf frame

        self.cam1_camera_model = 'pinhole'
        self.cam1_distortion_model = 'radtan'
        self.cam1_distortion_coeffs = np.array(
            [-0.361369, 0.182933, 0.0008665, 0.001038, -0.067140]) # wf frame
        # self.cam1_distortion_coeffs = np.array(
        #     [-0.217226, 0.172507, -0.000023, -0.001546, 1.958482]) # nf frame

        self.cam1_intrinsics = np.array([493.397, 499.81, 480, 270]) # wf frame
        # self.cam1_intrinsics = np.array([1822.5, 1817, 480, 270]) # nf frame

        self.cam1_resolution = np.array([960, 540])

        self.T_imu_body = np.identity(4)