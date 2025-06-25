from queue import Queue
from threading import Thread
import numpy as np

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF
from utils import to_quaternion
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



class VIO(object):
    def __init__(self, config, img_queue, imu_queue, gt_queue, viewer=None):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.gt_queue = gt_queue

        self.body_positions = {}
        self.body_quaternions = {}
        self.gt_positions = {}


        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.gt_thread = Thread(target=self.process_groundtruth)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.gt_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            print('imu_msg', imu_msg.timestamp)
            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)
            

            
    def process_groundtruth(self):
        while True:
            gt_msg = self.gt_queue.get()
            if gt_msg is None:
                print('gt_len:', len(self.gt_positions))
                return
            print('gt_msg', gt_msg.timestamp)
            print(gt_msg.p)
            if gt_msg.timestamp not in self.gt_positions:
                self.gt_positions[gt_msg.timestamp] = gt_msg.p
                

        # # Compute RMSE
        # rmse = self.compute_rmse()
        # print(rmse) 

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)
            print('result', result)

            if result is not None:
                if self.viewer is not None:
                    self.viewer.update_pose(result.cam0_pose)
                if feature_msg.timestamp not in self.body_positions:
                    self.body_positions[feature_msg.timestamp] = result.pose.t
                    quat = to_quaternion(result.pose.R)
                    self.body_quaternions[feature_msg.timestamp] = quat
                    print('body_positions', result.pose.t, 'quat', quat)
        


if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher
    # from viewer import Viewer

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='path/to/your/EuRoC_MAV_dataset/MH_01_easy', 
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    args = parser.parse_args()

    if args.view:
        viewer = Viewer()
    else:
        viewer = None

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=102.0)   # start from static state


    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue, gt_queue, viewer=viewer)


    # duration = float('inf')  # chạy hết sample
    duration = 120
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)
    gt_publisher = DataPublisher(
        dataset.groundtruth, gt_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    gt_publisher.start(now)

    msckf_vio.img_thread.join()
    msckf_vio.imu_thread.join()
    msckf_vio.gt_thread.join()
    msckf_vio.vio_thread.join()

    # Hàm ghép timestamp gần nhất giữa hai tập
    def match_positions(est_dict, gt_dict, threshold=0.01, output_csv='data_raw/matched_positions.csv'):
        est_times = sorted(est_dict.keys())
        gt_times = sorted(gt_dict.keys())
        
        est_matched = []
        gt_matched = []
        matched_timestamps = []

        j = 0
        for est_t in est_times:
            # Tìm gt_t gần nhất với est_t
            while j + 1 < len(gt_times) and abs(gt_times[j + 1] - est_t) < abs(gt_times[j] - est_t):
                j += 1
            if abs(gt_times[j] - est_t) < threshold:
                est_pos = est_dict[est_t]
                gt_pos = gt_dict[gt_times[j]]
                est_matched.append(est_pos)
                gt_matched.append(gt_pos)
                matched_timestamps.append((est_t, gt_times[j]))

        # Lưu vào CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['est_time', 'gt_time', 'est_x', 'est_y', 'est_z', 'gt_x', 'gt_y'])
            for (est_t, gt_t), est_pos, gt_pos in zip(matched_timestamps, est_matched, gt_matched):
                writer.writerow([
                    est_t, gt_t,
                    est_pos[0], est_pos[1], est_pos[2],
                    gt_pos[0], gt_pos[1]
                ])

        return np.array(est_matched), np.array(gt_matched)

    # Sử dụng hàm này để đồng bộ và tính RMSE
    print('IMU len:', len(msckf_vio.body_positions))
    print('GT len:', len(msckf_vio.gt_positions))
    imu_pos, gt_pos = match_positions(msckf_vio.body_positions, msckf_vio.gt_positions, threshold=0.01)
    print(f"Số cặp timestamp ghép được: {len(imu_pos)}")
    print("Imu:", imu_pos)
    print("GT:", gt_pos)

    def kabsch_umeyama(A, B, scale=True):
        A = np.asarray(A)
        B = np.asarray(B)
        assert A.shape == B.shape
        n, m = A.shape

        EA = np.mean(A, axis=0)
        EB = np.mean(B, axis=0)
        VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])
        R = U @ S @ VT
        c = VarA / np.trace(np.diag(D) @ S)
        if not scale:
            c = 1.0
        t = EA - c * R @ EB

        align = np.array([t + c * R @ b for b in B])

        return align, R, c, t

    def compute_rmse(est_positions, gt_positions):
        est_positions = np.asarray(est_positions)[:, :2]
        gt_positions = np.asarray(gt_positions)
        assert est_positions.shape == gt_positions.shape, "Shape mismatch!"
        # Tính khoảng cách Euclidean tại mỗi thời điểm
        errors = np.linalg.norm(est_positions - gt_positions, axis=1)

        # Tính RMSE dựa trên các lỗi Euclidean này
        rmse = np.sqrt(np.mean(errors ** 2))
        return rmse
    
    aligned_pos, R, scale, t= kabsch_umeyama(gt_pos, imu_pos[:, :2], scale=False)

    rmse = compute_rmse(aligned_pos, gt_pos)
    # rmse = compute_rmse(imu_pos, gt_pos)
    print(f"RMSE: {rmse:.4f} m")


    # === Visualization ===
    plt.figure(figsize=(8,6))
    plt.plot(gt_pos[:,0], gt_pos[:,1], 'go-', label='Ground Truth')
    plt.plot(imu_pos[:,0], imu_pos[:,1], 'r.--', label='Estimated (raw)')
    plt.plot(aligned_pos[:,0], aligned_pos[:,1], 'b.-', label='Estimated (aligned)')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Trajectory Alignment using Horn’s Method (Umeyama)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('trajectory.png')
    # plt.show()


    # with open('/home/huynt119/Documents/Github/rpg_trajectory_evaluation/results/euroc_mono_stereo/laptop/vio_stereo/laptop_vio_stereo_V2_01/msckf_stamped_traj_estimate.txt', 'w') as f:
    #     f.write("# time x y z qx qy qz qw\n")
    #     for t in sorted(msckf_vio.body_positions.keys()):
    #         if t in msckf_vio.body_quaternions:
    #             tx, ty, tz = msckf_vio.body_positions[t]
    #             qx, qy, qz, qw = msckf_vio.body_quaternions[t]
    #             f.write(f"{t} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")