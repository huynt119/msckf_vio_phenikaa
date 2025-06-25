import numpy as np
import cv2
import os
import time
import csv

from collections import defaultdict, namedtuple

from threading import Thread



class GroundTruthReader(object):
    def __init__(self, path, scaler, starttime=-float('inf')):
        self.scaler = scaler   # convert timestamp from ns to second
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('gt_msg', ['timestamp', 'p', 'q'])

    def parse(self, line):
        """
        line: timestamp, x, y, roll, pitch, yaw, lat, lon, qx, qy, qz, qw
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        p = np.array(line[1:3]) # position (x, y)
        q = np.array(line[8:12])  # quaternion (qx, qy, qz, qw)
        return self.field(timestamp, p, q)

    def set_starttime(self, starttime):
        self.starttime = starttime

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data



class IMUDataReader(object):
    def __init__(self, path, scaler, starttime=-float('inf')):
        self.scaler = scaler
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('imu_msg', 
            ['timestamp', 'angular_velocity', 'linear_acceleration'])

    def parse(self, line):
        """
        line:  IMUtimestamp, angRateXRaw, angRateYRaw, angRateZRaw, accXRaw, accYRaw, accZRaw
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        wm = np.array(line[1:4])
        am = np.array(line[4:7])
        return self.field(timestamp, wm, am)

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        # return next(self).timestamp
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime



class ImageReader(object):
    def __init__(self, ids, timestamps, starttime=-float('inf')):
        self.ids = ids
        self.timestamps = timestamps
        self.starttime = starttime
        self.cache = dict()
        self.idx = 0

        self.field = namedtuple('img_msg', ['timestamp', 'image'])

        self.ahead = 10   # 10 images ahead of current index
        self.wait = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    # def read(self, path):
    #     return cv2.imread(path, -1)

    def read(self, path):
        img = cv2.imread(path, -1)
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.wait:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if self.timestamps[i] < self.starttime:
                    continue
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            if timestamp < self.starttime:
                continue
            yield self.field(timestamp, self[i])

    def start_time(self):
        return min(self.timestamps)

    def set_starttime(self, starttime):
        self.starttime = starttime



class Stereo(object):
    def __init__(self, cam0, cam1):
        assert len(cam0) == len(cam1)
        self.cam0 = cam0
        self.cam1 = cam1
        self.timestamps = cam0.timestamps

        self.field = namedtuple('stereo_msg', 
            ['timestamp', 'cam0_image', 'cam1_image', 'cam0_msg', 'cam1_msg'])

    def __iter__(self):
        for l, r in zip(self.cam0, self.cam1):
            assert abs(l.timestamp - r.timestamp) < 0.01, 'unsynced stereo pair'
            yield self.field(l.timestamp, l.image, r.image, l, r)

    def __len__(self):
        return len(self.cam0)

    def start_time(self):
        return self.cam0.starttime

    def set_starttime(self, starttime):
        self.starttime = starttime
        self.cam0.set_starttime(starttime)
        self.cam1.set_starttime(starttime)
        
    

class EuRoCDataset(object):   # Stereo + IMU
    '''
    path example: 'path/to/your/EuRoC Mav Dataset/MH_01_easy'
    '''
    def __init__(self, path):
        self.path = path
        self.groundtruth = GroundTruthReader(os.path.join(
            path, 'ground_truth.csv'), 1e-3)
        self.imu = IMUDataReader(os.path.join(
            path, 'imu.csv'), 1e-3)
        self.cam0 = ImageReader(
            *self.list_imgs(os.path.join(path, 'mf_wf.csv'), 'mf')) # Đọc dữ liệu align timestamp giữa 2 cam
        self.cam1 = ImageReader(
            *self.list_imgs(os.path.join(path, 'wf_mf.csv'), 'wf'))

        self.stereo = Stereo(self.cam0, self.cam1)
        self.timestamps = self.cam0.timestamps

        self.starttime = max(self.imu.start_time(), self.stereo.start_time())
        self.set_starttime(0)

    def set_starttime(self, offset):
        self.groundtruth.set_starttime(self.starttime + offset)
        self.imu.set_starttime(self.starttime + offset)
        self.cam0.set_starttime(self.starttime + offset)
        self.cam1.set_starttime(self.starttime + offset)
        self.stereo.set_starttime(self.starttime + offset)

    # def list_imgs(self, dir):
    #     xs = [_ for _ in os.listdir(dir) if _.endswith('.jpg')]
    #     xs = sorted(xs, key=lambda x:float(x[:-4]))
    #     timestamps = [float(_[:-4]) * 1e-9 for _ in xs]
    #     return [os.path.join(dir, _) for _ in xs], timestamps

    def list_imgs(self, csv_path, sub_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]

        # Sắp xếp theo timestamp tăng dần
        rows.sort(key=lambda row: float(row['timestamp']))

        timestamps = [float(row['timestamp']) * 1e-3 for row in rows]
        image_paths = [os.path.join(self.path, sub_path, row['filename']) for row in rows]
        return image_paths, timestamps




# simulate the online environment
class DataPublisher(object):
    def __init__(self, dataset, out_queue, duration=float('inf'), ratio=1.): 
        self.dataset = dataset
        self.dataset_starttime = dataset.starttime
        self.out_queue = out_queue
        self.duration = duration
        self.ratio = ratio
        self.starttime = None
        self.started = False
        self.stopped = False

        self.publish_thread = Thread(target=self.publish)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()

    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
        self.out_queue.put(None)

    def publish(self):
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                data = next(dataset)
            except StopIteration:
                self.out_queue.put(None)
                return

            interval = data.timestamp - self.dataset_starttime
            if interval < 0:
                continue
            while (time.time() - self.starttime) * self.ratio < interval + 1e-3:
                time.sleep(1e-3)   # assumption: data frequency < 1000hz
                if self.stopped:
                    return

            if interval <= self.duration + 1e-3:
                self.out_queue.put(data)
            else:
                self.out_queue.put(None)
                return



if __name__ == '__main__':
    from queue import Queue

    path = 'data_raw'
    dataset = EuRoCDataset(path)
    dataset.set_starttime(offset=0)

    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()

    duration = 1
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration)
    gt_publisher = DataPublisher(
        dataset.groundtruth, gt_queue, duration)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    gt_publisher.start(now)

    def print_msg(in_queue, source):
        while True:
            x = in_queue.get()
            if x is None:
                return
            print(x.timestamp, source)
    t2 = Thread(target=print_msg, args=(imu_queue, 'imu'))
    t3 = Thread(target=print_msg, args=(gt_queue, 'groundtruth'))
    t2.start()
    t3.start()
    timestamps = []
    while True:
        x = img_queue.get()
        if x is None:
            break
        print(x.timestamp, 'image')
        # cv2.imshow('left', np.hstack([x.cam0_image, x.cam1_image]))
        # cv2.waitKey(1)
        timestamps.append(x.timestamp)

    imu_publisher.stop()
    img_publisher.stop()
    gt_publisher.stop()
    t2.join()
    t3.join()

    print(f'\nelapsed time: {time.time() - now}s')
    print(f'dataset time interval: {timestamps[-1]} -> {timestamps[0]}'
        f'  ({timestamps[-1]-timestamps[0]}s)\n')
    print('Please check if IMU and image are synced')


