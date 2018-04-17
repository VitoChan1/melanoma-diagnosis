import os
import numpy as np
from scipy.misc import imsave
import matplotlib.pylab as plt


data_dir = './ISBI2016_ISIC_Part3B_Training_Data'
cropped_data_dir = data_dir + '_tight_cropped/'
if not os.path.exists(cropped_data_dir):
    os.makedirs(cropped_data_dir)


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def main():
    data_filenames = sorted(os.listdir(data_dir))
    img_filenames = data_filenames[0::2]
    seg_filenames = data_filenames[1::2]

    for img_filename, seg_filename in zip(img_filenames, seg_filenames):
        img = plt.imread(data_dir + '/' + img_filename)
        seg = plt.imread(data_dir + '/' + seg_filename)
        rmin, rmax, cmin, cmax = bbox(seg)

        imsave(cropped_data_dir + img_filename,
               img[rmin:rmax + 1, cmin:cmax + 1])


if __name__ == '__main__':
    main()
