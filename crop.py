import os
import math
import numpy as np
from scipy.misc import imsave
import matplotlib.pylab as plt


data_dir = './ISBI2016_ISIC_Part1_Training_Data'
label_dir = './ISBI2016_ISIC_Part1_Training_GroundTruth'

cropped_data_dir = data_dir + '_cropped/'
cropped_label_dir = label_dir + '_cropped/'
if not os.path.exists(cropped_data_dir):
    os.makedirs(cropped_data_dir)
if not os.path.exists(cropped_label_dir):
    os.makedirs(cropped_label_dir)


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def main():
    data_filenames = sorted(os.listdir(data_dir))
    label_filenames = sorted(os.listdir(label_dir))

    for data_file, label_file in zip(data_filenames, label_filenames):
        data = plt.imread(data_dir + '/' + data_file)
        label = plt.imread(label_dir + '/' + label_file)
        rmin, rmax, cmin, cmax = bbox(label)
        rcenter = (rmin + rmax) / 2
        ccenter = (cmin + cmax) / 2
        height = rmax - rmin + 1
        width = cmax - cmin + 1

        if 1.1 * height > 1.3 * width:
            dst_height = int(math.ceil(1.1 * height))
            dst_width = int(math.ceil(1.3 * width))
        elif 1.3 * height < 1.1 * width:
            dst_height = int(math.ceil(1.3 * height))
            dst_width = int(math.ceil(1.1 * width))
        else:
            dst_height = int(math.ceil(1.2 * height))
            dst_width = int(math.ceil(1.2 * width))
        dst_height = max(dst_height, 480)
        dst_width = max(dst_width, 480)

        rmin = rcenter - dst_height / 2
        rmax = rmin + dst_height
        cmin = ccenter - dst_width / 2
        cmax = cmin + dst_width

        if rmin < 0:
            rmin = 0
            rmax = rmax - rmin
        elif rmax > data.shape[0]:
            rmax = data.shape[0]
            rmin = rmin - rmax + data.shape[0]

        if cmin < 0:
            cmin = 0
            cmax = cmax - cmin
        elif cmax > data.shape[1]:
            cmax = data.shape[1]
            cmin = cmin - cmax + data.shape[1]

        imsave(cropped_data_dir + data_file, data[rmin:rmax, cmin:cmax])
        imsave(cropped_label_dir + label_file,
               label[rmin:rmax, cmin:cmax])




if __name__ == '__main__':
    main()
