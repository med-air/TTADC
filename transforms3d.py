import numpy as np
import numbers
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import (
    rotate,
    zoom
)


# class NormalizeInstance3D(object):
#     """Normalize a tensor volume with mean and standard deviation estimated
#     from the sample itself.

#     :param mean: mean value.
#     :param std: standard deviation value.
#     """

#     def __call__(self, sample):

#         mean, std = sample.mean(), sample.std()

#         if mean != 0 or std != 0:
#             input_data_normalized = F.normalize(input_data,
#                                                 [mean for _ in range(
#                                                     0, input_data.shape[0])],
#                                                 [std for _ in range(0, input_data.shape[0])])
#         return sample


class RandomRotation3D(object):
    """Make a rotation of the volume's values.

    :param degrees: Maximum rotation's degrees.
    :param axis: Axis of the rotation.
    """

    def __init__(self, degrees, axis=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.axis = axis

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        sample = sample.squeeze()
        if len(sample.shape) != 3:
            raise ValueError(
                "Input of RandomRotation3D should be a 3 dimensionnal tensor.")

        angle = RandomRotation3D.get_params(self.degrees)
        axes = [0, 1, 2]
        axes.remove(self.axis)
        return np.array([rotate(sample, angle, axes=axes, reshape=False)])


class RandomShift3D(object):
    """Make a shifting of selected axes.
    shift_range and axes must be paired, that is len(shift_range) == len(axes)
    """

    def __init__(self, shift_range=(10, 10, 10), axes=(0, 1, 2)):
        self.axes = axes
        self.shift_range = shift_range
        assert len(shift_range) == len(axes)

    @staticmethod
    def get_params(shift_range=(10, 10, 10), axes=(0, 1, 2)):
        paddings = []
        for shift, axis in zip(shift_range, axes):
            shifting = np.random.randint(-shift, shift)
            padding = [(0, 0), (0, 0), (0, 0)]
            padding[axis] = (-shifting, 0) if shifting < 0 else (0, shifting)
            paddings.append((axis, shifting, padding))
        return paddings

    def __call__(self, sample):
        input_data = sample.squeeze()
        params = RandomShift3D.get_params(self.shift_range, self.axes)
        for axis, shifting, padding in params:
            input_data = np.pad(input_data, padding, mode='constant')
            indices = np.arange(0, sample.squeeze().shape[axis]) if shifting < 0 else np.arange(shifting, input_data.shape[axis])
            input_data = np.take(input_data, indices, axis=axis)
        return np.array([input_data])


class RandomFlip3D(object):

    def __init__(self, axes=(0, 1, 2), p=0.5):
        self.axes = axes
        if (type(p) == list or type(p) == tuple) and len(p) == len(axes):
            self.p = list(p)
        elif type(p) == float:
            self.p = [p] * len(axes)
        else:
            raise ValueError()

    @staticmethod
    def get_params(axes=(0, 1, 2), p=0.5):
        params = []
        for axis, p in zip(axes, p):
            if np.random.random() < p:
                params.append([axis, True])
            else:
                params.append([axis, False])
        return params

    def __call__(self, sample):
        input_data = sample.squeeze()
        params = RandomFlip3D.get_params(self.axes, self.p)
        for axis, p in params:
            if p:
                input_data = np.flip(input_data, axis=axis).copy()
        return np.array([input_data])


class GaussianDenoising(object):

    def __init__(self, sigma_range=(0, 2)):
        self.sigma_range = sigma_range

    @staticmethod
    def get_params(sigma_range=(0, 2)):
        return np.random.random() * np.random.randint(*sigma_range)

    def __call__(self, sample):
        input_data = sample.squeeze()
        sigma = GaussianDenoising.get_params(self.sigma_range)
        input_data = gaussian_filter(input_data, sigma)
        return np.array([input_data])


class ElasticTransform(object):
    def __init__(self, alpha_range, sigma_range,
                 p=0.5, labeled=True):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.labeled = labeled
        self.p = p

    @staticmethod
    def get_params(alpha, sigma):
        alpha = np.random.uniform(alpha[0], alpha[1])
        sigma = np.random.uniform(sigma[0], sigma[1])
        return alpha, sigma

    @staticmethod
    def elastic_transform(image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]),
                           np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

    def sample_augment(self, input_data, params):
        param_alpha, param_sigma = params

        np_input_data = np.array(input_data)
        np_input_data = self.elastic_transform(np_input_data,
                                               param_alpha, param_sigma)
        input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def label_augment(self, gt_data, params):
        param_alpha, param_sigma = params

        np_gt_data = np.array(gt_data)
        np_gt_data = self.elastic_transform(np_gt_data,
                                            param_alpha, param_sigma)
        np_gt_data[np_gt_data >= 0.5] = 1.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        gt_data = Image.fromarray(np_gt_data, mode='F')
        return gt_data

    def __call__(self, sample):
        if np.random.random() < self.p:
            input_data = sample.squeeze()
            params = self.get_params(self.alpha_range,
                                     self.sigma_range)
            if isinstance(input_data, list):
                ret_input = [self.sample_augment(item, params)
                             for item in input_data]
            else:
                ret_input = self.sample_augment(input_data, params)

        return np.array([ret_input])


class AdditiveGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rdict = {}
        input_data = sample.squeeze()

        noise = np.random.normal(self.mean, self.std, input_data.size)
        noise = noise.astype(np.float32)

        np_input_data = np.array(input_data)
        np_input_data += noise
        input_data = Image.fromarray(np_input_data, mode='F')
        return np.array([input_data])


class HistogramClipping(object):
    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, sample):
        array = np.copy(sample)
        percentile1 = np.percentile(array, self.min_percentile)
        percentile2 = np.percentile(array, self.max_percentile)
        array[array <= percentile1] = percentile1
        array[array >= percentile2] = percentile2
        return array
