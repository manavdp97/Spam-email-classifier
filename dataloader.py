import struct
import numpy as np


class DataLoader:
    def __init__(self, file_images, file_labels):
        with open(file_images, 'rb') as f:
            _, num, rows, cols = struct('>IIII', f.read(16))
            train_data = np.fromfile(f, dtype=np.uint8)