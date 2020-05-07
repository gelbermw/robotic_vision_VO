import numpy as np
import random
import os
import pickle


def normalize(data, max_value=1, min_value=0):
    data = np.abs(data)
    mx = np.max(data)
    mn = np.min(data)
    mx = max(mx, abs(mn))
    mn = 0
    return (data - mn) / (mx - mn) * (max_value - min_value) + min_value


def normalize_full(data, max_value=1, min_value=0):
    # data = np.abs(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - mn) / (mx - mn) * (max_value - min_value) + min_value


def load_data(filename):
    dimensions = (144, 176)
    with open(filename, 'r') as f:
        data = f.read().split()
        values = []
        for elem in data:
            try:
                values.append(float(elem))
            except ValueError:
                pass

        values = np.array(values)
        values = np.reshape(values, (-1, dimensions[1]))

        z = np.reshape(values[0:144], dimensions)       # z
        x = np.reshape(values[144:288], dimensions)        #x
        y = np.reshape(values[288:432], dimensions)         #y
        amp = np.reshape(values[432:576], dimensions)
        conf = np.reshape(values[576:720], dimensions)
        time = np.reshape(values[720], dimensions[1])

        return DataEntry(x, y, z, amp, conf, time)


class DataEntry:

    def __init__(self, x, y, z, amplitude, conf, times):
        self.x = x
        self.y = y
        self.z = z
        self.amplitude = np.sqrt(amplitude)
        self.confidence = conf
        self.timestamps = times



    def get_z_image(self):
        return np.float32(normalize(self.z))

    def get_x_image(self):
        return np.float32(normalize(self.x))

    def get_y_image(self):
        return np.float32(normalize(self.y))

    def get_amplitude_image(self):
        return np.float32(normalize(self.amplitude, 1, 0))

    def get_confidence_image(self):
        return np.float32(normalize(self.confidence))

    def get_position(self, px, py):
        px = int(px)
        py = int(py)
        return self.z[py][px], self.x[py][px], self.y[py][px]

    def get_position_(self, pos):
        return self.get_position(pos[0], pos[1])

    def get_position_set(self):
        return np.array((self.z, self.x, self.y)).T

    def get_confidence(self, px, py):

        c1 = self.confidence[int(py)][int(px)]
        # c2 = self.confidence[int(py+1)][int(px)]
        # c3 = self.confidence[int(py)][int(px+1)]
        # c4 = self.confidence[int(py-1)][int(px)]
        # c5 = self.confidence[int(py)][int(px-1)]

        return c1# min(c1, c2, c3, c4, c5)

    def get_combined_image(self):
        x = self.get_x_image()
        y = self.get_y_image()
        z = self.get_z_image()
        conf = self.get_amplitude_image()

        disp = np.hstack((x, y))
        d2 = np.hstack((z, conf))
        return np.vstack((disp, d2))


class Dataset:


    def __init__(self, dir_set, pkl_dir='./data/', num_images=-1):
        self.directory = dir_set[0]
        self.name = dir_set[1]
        self.num_images = num_images
        self.data = []

        self.rand_idx = 0

        pkl_file = os.path.join(pkl_dir, '{}.pkl'.format(self.name))
        if not os.path.exists(pkl_file):
            self.data = self.load_all()
            pickle.dump(self.data, open(pkl_file, 'wb'))
        else:
            self.data = pickle.load(open(pkl_file, 'rb'))

        self.num_images = len(self.data)

    def load_all(self):
        self.data = []

        files = os.listdir(self.directory)
        print('Loading %d files from %s' % (len(files), self.directory))
        files.sort()

        count = 0

        for file in files:
            full_path = os.path.join(self.directory, file)
            self.data.append(load_data(full_path))
            if 0 < self.num_images < count:
                break
            count += 1

        return self.data

    def is_first_entry(self):
        return self.rand_idx == 0

    def next_entry(self):
        ret = self.data[self.rand_idx]
        self.rand_idx += 1
        if self.rand_idx >= self.num_images:
            self.rand_idx = 0
        return ret

    def random_entry(self):
        return random.choice(self.data)
