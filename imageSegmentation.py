import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
from skimage import color
import time
import os
import scipy.io as sio


def findpeak(data, idx, threshold, r):
    """
    Implements the peak searching process of the algorithm.

    :param data: image data in the format [number of pixels]x[feature vector].
    :param idx: the index of the point to calculate its corresponding density peak.
    :param threshold: a low number that tells when the shift ends.
    :param r: the radius of the search window.
    :return: peak
    """
    point = data[idx, :]
    point = point.T.reshape((1, 3))
    shift = np.ones_like(point)
    while shift.all() > threshold:
        distances = cdist(data, point, metric='euclidean')
        found_inds = np.argwhere(distances <= r)[:, 0]
        found = data[found_inds, ]
        peak = np.mean(found, axis=0)
        shift = abs(point - peak)
        point = peak.reshape((1, 3))

    return peak.reshape(1, peak.shape[0])


def meanshift(data, r, threshold):
    """
    Clusters the dataset by associating each point to a peak of the dataset's
    probability density.

    :param data: image data in the format [number of pixels]x[feature vector].
    :param r: the radius of the search window.
    :param threshold: a low number that tells when the shift ends.
    :return: labels, peaks
    """

    labels = np.zeros_like(data[:, 0])
    peaks = []
    n_labels = 1
    peak = findpeak(data, 0, threshold, r)
    peaks = np.append(peaks, peak)
    peaks = peaks.reshape(1, 3)
    labels[0] = n_labels
    for idx in tqdm(range(len(data))):
        peak = findpeak(data, idx, threshold, r)
        dif = cdist(peaks, peak, 'euclidean')
        if np.any(dif < r/2):
            labels[idx] = n_labels
        else:
            n_labels += 1
            labels[idx] = n_labels
            peaks = np.append(peaks, peak, axis=0)

    return labels, peaks


def meanshift_opt(data, r, threshold, c):
    """
    Clusters the dataset by associating each point to a peak of the dataset's
    probability density. Optimized function.

    :param data: image data in the format [number of pixels]x[feature vector].
    :param r: the radius of the search window.
    :param threshold: a low number that tells when the shift ends.
    :param c: constant value
    :return: labels, peaks
    """
    labels = np.zeros_like(data[:, 0])
    peaks = []
    n_labels = 1
    peak, cpts = findpeak_opt(data, 0, r, threshold, c)
    peaks = np.append(peaks, peak)
    peaks = peaks.reshape(1, 3)
    cpts_peaks = np.where(cpts == 1)
    labels[cpts_peaks] = n_labels

    for idx in tqdm(range(len(data))):
        flag = True
        if labels[idx] > 0:
            continue
        peak, cpts = findpeak_opt(data, idx, r, threshold, c)
        cpts_peaks = np.where(cpts == 1)
        for idx2 in range(n_labels):
            dif = cdist(peaks[idx2].reshape(1, 3), peak, metric='euclidean')
            if np.any(dif < r):
                labels[cpts_peaks] = idx2 + 1
                flag = False

        if flag:
            n_labels += 1
            labels[cpts_peaks] = n_labels
            peaks = np.append(peaks, peak, axis=0)
    return labels, peaks


def findpeak_opt(data, idx, r, threshold, c=4):
    """
    Implements the peak searching process of the algorithm. Optimized function.

    :param data: image data in the format [number of pixels]x[feature vector].
    :param idx: the index of the point to calculate its corresponding density peak.
    :param r: the radius of the search window.
    :param threshold: a low number that tells when the shift ends.
    :param c: constant value
    :return: peak, cpts
    """
    point = data[idx, :]
    point = point.T.reshape((1, 3))
    shift = np.ones_like(point)
    cpts = np.zeros_like(data[:, 0])
    while shift.all() > threshold:
        distances = cdist(data, point, metric='euclidean')
        found_inds = np.argwhere(distances <= r)[:, 0]
        found = data[found_inds, ]
        found_inds = np.argwhere(distances <= r/c)[:, 0]
        cpts[found_inds] = 1
        peak = np.mean(found, axis=0)
        shift = abs(point - peak)
        point = peak.reshape((1, 3))

    return peak.reshape(1, peak.shape[0]), cpts


def segmIm(img, r, threshold=0.01, c=4):
    """
        Implements the actual segmentation.

    :param img: image in the RGB color space
    :param r: the radius of the search window.
    :param threshold: a low number that tells when the shift ends.
    :param c: constant value
    :return: segmented image
    """

    img_CEL = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    img_CEL = color.rgb2lab(img_CEL)
    img_res = np.zeros_like(img_CEL)

    labels, peaks = meanshift_opt(img_CEL, r, threshold, c)
    labels = labels.reshape(labels.shape[0], 1).astype(int)
    length = peaks.shape[0]
    for index in range(length):
        inds = np.where(labels == index + 1)
        img_res[inds[0], :] = peaks[index]
    plotclustersimage3D(img.reshape(img.shape[0]*img.shape[1], img.shape[2]), labels, peaks)
    img_res = img_res.reshape(img.shape[0], img.shape[1], img.shape[2])
    img_res = color.lab2rgb(img_res)
    return img_res


def plotclusters3D(data, labels, peaks):
    """
    Plots the modes of the given data set in 3D by coloring each pixel
    and each cluster peak.

    Args:
        data: dataset in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    plt.figure()
    ax = plt.axes(projection='3d')
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[..., ::]
    for idx, peak in enumerate(rgb_peaks):
        cluster = data[np.where(labels == idx+1)[0]].T
        ax.scatter(peak[0], peak[1], peak[2], color='b', marker='H', s=100)
        ax.scatter(cluster[0], cluster[1], cluster[2], color='r', marker='*', s=.5, alpha=0.3)
    plt.show()


def plotclustersimage3D(data, labels, peaks):
    """
    Plots the modes of the given image data in 3D by coloring each pixel.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[..., ::-1]
    rgb_peaks /= 255.0
    colors = []
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        while color in colors:
            color = np.random.uniform(0, 1, 3)
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    fig.show()


def main(argv):
    '''
    main function of the file

    :param argv[0]: the name of the image
    :param argv[1]: the radius
    '''

    image_name = argv[0]
    r = argv[1]
    image_path = os.path.join(os.path.curdir, image_name)
    img = cv2.imread(image_path)
    c = 4
    threshold = 0.01
    img_copy = img.copy()
    start = time.time()
    img_res = segmIm(img_copy, r, threshold, c)
    end = time.time()
    print("The algorithm executed in", (end-start))
    cv2.imshow('Segmented Image', img_res)
    cv2.waitKey()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
