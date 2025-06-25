import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # cần import để vẽ 3D

def read_traj_file(path):
    """
    Đọc file trajectory .txt với format: # time x y z ...
    Trả về: dict {timestamp: np.array([x, y, z])}
    """
    traj = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            if ',' in line:
                parts = line.strip().split(',')
            else:
                parts = line.strip().split() 
            if len(parts) < 4:
                continue
            t = float(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            traj[t] = xyz
    return traj

def match_positions(est_dict, gt_dict, threshold=0.01, output_csv='matched_positions.csv'):
    est_times = sorted(est_dict.keys())
    gt_times = sorted(gt_dict.keys())
    
    est_matched = []
    gt_matched = []
    matched_timestamps = []
    print(len(est_times), len(gt_times))

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
        writer.writerow(['est_time', 'gt_time', 'est_x', 'est_y', 'est_z', 'gt_x', 'gt_y', 'gt_z'])
        for (est_t, gt_t), est_pos, gt_pos in zip(matched_timestamps, est_matched, gt_matched):
            writer.writerow([
                est_t, gt_t,
                est_pos[0], est_pos[1], est_pos[2],
                gt_pos[0], gt_pos[1], gt_pos[2]
            ])

    return np.array(est_matched), np.array(gt_matched)

def compute_rmse(est_positions, gt_positions):
    est_positions = np.asarray(est_positions)
    gt_positions = np.asarray(gt_positions)
    assert est_positions.shape == gt_positions.shape, "Shape mismatch!"

    # Tính khoảng cách Euclidean tại mỗi thời điểm
    errors = np.linalg.norm(est_positions - gt_positions, axis=1)

    # Tính RMSE dựa trên các lỗi Euclidean này
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse


# Hàm align trajectory
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--est', type=str, required=True, help='Path to estimated trajectory txt')
    parser.add_argument('--gt', type=str, required=True, help='Path to groundtruth trajectory txt')
    parser.add_argument('--threshold', type=float, default=0.01, help='Time sync threshold (s)')
    parser.add_argument('--plot_3d', action='store_true', help='Plot in 3D instead of 2D')
    args = parser.parse_args()

    est_traj = read_traj_file(args.est)
    gt_traj = read_traj_file(args.gt)

    est_pos, gt_pos = match_positions(est_traj, gt_traj, threshold=args.threshold, output_csv='/home/huynt119/Documents/Github/stereo_msckf/matched_positions.csv')
    print(f"Số cặp timestamp ghép được: {len(est_pos)}")
    aligned_pos, R, c, t = kabsch_umeyama(gt_pos, est_pos, scale=False)
    print(R, c, t)
    # rmse = compute_rmse(est_pos, gt_pos)
    rmse = compute_rmse(aligned_pos, gt_pos)
    print(f"RMSE: {rmse:.6f} m")




    if args.plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'go-', label='Ground Truth')
        ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r.--', label='Estimated (raw)')
        ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2], 'b.-', label='Estimated (aligned)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Trajectory Alignment (3D) using Horn’s Method')
        ax.legend()
        plt.tight_layout()
        plt.savefig('trajectory_eval_3d.png')
    else:
        plt.figure(figsize=(8, 6))
        plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'go-', label='Ground Truth')
        plt.plot(est_pos[:, 0], est_pos[:, 1], 'r.--', label='Estimated (raw)')
        plt.plot(aligned_pos[:, 0], aligned_pos[:, 1], 'b.-', label='Estimated (aligned)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Trajectory Alignment (2D) using Horn’s Method')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig('trajectory_eval.png')