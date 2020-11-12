from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
# import collections
import numpy as np
# from sklearn.utils import shuffle
import os
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
class InputIterator(object):
    def __init__(
        self,
        batch_size=32,
        images_directory='/home/bartheq/projects/kanazawa/data/images/train/NG',
    ):
        self.batch_size = batch_size
        self.images_directory = images_directory
        self.images_dict = dict()
        for folder in os.listdir(self.images_directory):
            folder += '/color'
            self.images_dict[folder] = os.listdir(
                os.path.join(self.images_directory, folder))
        self.labels_map = dict()
        for i, folder in enumerate(self.images_dict.keys()):
            self.labels_map[i] = folder
        self.image_paths, self.labels = [], []
        for i, folder in enumerate(self.images_dict.keys()):
            for item in self.images_dict[folder]:
                self.image_paths.append(item)
                self.labels.append(i)
        self.data_set_len = len(self.labels)
        device_id = 0
        self.labels = self.labels[self.data_set_len *
                                  device_id:self.data_set_len *
                                  (device_id + 1)]
        self.n = len(self.labels)
#         self.image_paths, self.labels = shuffle(self.image_paths, self.labels)
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        image_batch = []
        label_batch = []
        if self.i >= self.n:
            raise StopIteration
        for _ in range(self.batch_size):
            filename = os.path.join(self.images_directory,
                                    self.labels_map[self.labels[self.i]],
                                    self.image_paths[self.i])
            f = open(filename, 'rb')
            image_batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            label_batch.append(np.array(self.labels[self.i], dtype=np.uint8))
            self.i += 1
        return (image_batch, label_batch)
    @property
    def size(self):
        return self.data_set_len
    next = __next__
class NoAugDataPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        data_iterator,
        num_threads,
        device_id,
    ):
        super(NoAugDataPipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=44,
        )
        self.source = ops.ExternalSource()
        self.source_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device='gpu', output_type=types.RGB)
        self.transpose = ops.Transpose(device='gpu', perm=[2, 0, 1])
        self.external_data = data_iterator
        self.iterator = iter(self.external_data)
    def define_graph(self):
        self.images = self.source()
        self.labels = self.source_label()
        labels = self.labels
        outputs = self.decode(self.images)
        outputs = self.transpose(outputs)
        return (outputs, labels)
    def iter_setup(self):
        try:
            (images, labels) = next(self.iterator)
            self.feed_input(self.images, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration
class DataPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        data_iterator,
        num_threads,
        device_id,
        prefetch_queue_depth=4,
    ):
        super(DataPipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=44,
        )
        self.source = ops.ExternalSource()
        self.source_label = ops.ExternalSource()
        self.transform_source = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.external_data = data_iterator
        self.iterator = iter(self.external_data)
        self.coin = ops.CoinFlip(probability=0.8)
        self.cast_fp32 = ops.Cast(dtype=types.FLOAT)
        self.br_c = ops.BrightnessContrast(device="gpu")
        self.brightness_rng = ops.NormalDistribution(mean=1, stddev=0.4)
        self.contrast_rng = ops.Uniform(range=(0.95, 1.05))
        self.vflip = ops.Flip(device='gpu', vertical=1)
        self.hflip = ops.Flip(device='gpu', horizontal=1)
        self.normal_noise = ops.NormalDistribution(stddev=50,
                                                   shape=[224, 192, 3])
        self.cast = ops.Cast(device='gpu', dtype=types.RGB)
        self.anchor_rng = ops.Uniform(range=(0.1, 0.9), shape=2)
        self.erase = ops.Erase(device='gpu',
                               shape=(10, 10),
                               normalized_anchor=True,
                               normalized_shape=False,
                               axis_names='HW')
        self.shear = ops.WarpAffine(device='gpu',
                                    size=(224, 192),
                                    interp_type=types.INTERP_NN,
                                    fill_value=0)
        self.transpose = ops.Transpose(device='gpu', perm=[2, 0, 1])
    def iter_setup(self):
        try:
            (images, labels) = next(self.iterator)
            self.feed_input(self.images, images)
            self.feed_input(self.labels, labels)
            self.feed_input(self.transform,
                            gen_transforms(self.batch_size, random_transform))
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration
    def define_graph(self):
        self.transform = self.transform_source()
        self.images = self.source()
        self.labels = self.source_label()
        labels = self.labels
        outputs = self.decode(self.images)
        brightness_val = self.brightness_rng()
        contrast_val = self.contrast_rng()
        outputs = self.br_c(
            outputs,
            brightness=brightness_val,
            contrast=contrast_val,
        )
        outputs = self.vflip(outputs)
        outputs = self.hflip(outputs)
        noise = self.normal_noise()
        outputs = self.cast(outputs)
        anchors = self.anchor_rng()
        outputs = self.erase(outputs, anchor=anchors)
        outputs = self.shear(outputs, matrix=self.transform)
        return (outputs, labels)
def random_transform(index):
    dst_cx, dst_cy = (200, 200)
    src_cx, src_cy = (200, 200)
    # This function uses homogeneous coordinates - hence, 3x3 matrix
    # translate output coordinates to center defined by (dst_cx, dst_cy)
    t1 = np.array([[1, 0, -dst_cx], [0, 1, -dst_cy], [0, 0, 1]])
    def u():
        return np.random.uniform(-0.1, 0.1)
    # apply a randomized affine transform - uniform scaling + some random distortion
    m = np.array([[1 + u(), u(), 0], [u(), 1 + u(), 0], [0, 0, 1]])
    # translate input coordinates to center (src_cx, src_cy)
    t2 = np.array([[1, 0, src_cx], [0, 1, src_cy], [0, 0, 1]])
    # combine the transforms
    m = (np.matmul(t2, np.matmul(m, t1)))
    # remove the last row; it's not used by affine transform
    return m[0:2, 0:3]
def gen_transforms(batch_size, single_transform_fn):
    out = np.zeros([batch_size, 2, 3])
    for i in range(batch_size):
        out[i, :, :] = single_transform_fn(i)
    return out.astype(np.float32)
def main():
    iterator = InputIterator()
    pipe = DataPipeline(
        batch_size=32,
        data_iterator=iterator,
        num_threads=2,
        device_id=0,
    )
    pii = PyTorchIterator(pipe, fill_last_batch=False)
    epochs = 10
    for e in range(epochs):
        for i, data in enumerate(pii):
            print("epoch: {}, iter {}, real batch size: {}".format(
                e, i, len(data[0]["data"])))
        pii.reset()
    pass
if __name__ == '__main__':
    main()