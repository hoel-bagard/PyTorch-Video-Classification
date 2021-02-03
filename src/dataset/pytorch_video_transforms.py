import random

import cv2
import numpy as np
import torch


class VideoTransforms:
    class Crop(object):
        """ Crop the video. """

        def __init__(self, left: int = 0, right: int = 1, top: int = 0, bottom: int = 1):
            self.left = left
            self.right = right
            self.top = top
            self.bottom = bottom

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            cropped_video = np.empty(video[:, self.top:-self.bottom, self.left:-self.right].shape)
            cropped_video = video[:, self.top:-self.bottom, self.left:-self.right]

            return {"data": cropped_video, "label": label}

    class RandomCrop(object):
        """ Random crops image """

        def __init__(self, size_reduction_factor: int = 0.95):
            self.size = size_reduction_factor

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            cropped_video = np.empty((video.shape[0], int(video.shape[1]*self.size),
                                      int(video.shape[2]*self.size), video.shape[-1]))
            h = random.randint(0, int(video.shape[1]*(1-self.size))-1)
            w = random.randint(0, int(video.shape[2]*(1-self.size))-1)
            cropped_video = video[:, h:h+int(video.shape[1]*self.size), w:w+int(video.shape[2]*self.size)]

            return {"data": cropped_video, "label": label}

    class Resize(object):
        """ Resize the video to a given size. """

        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            resized_video = np.empty((len(video), self.height, self.width, video.shape[-1]))
            for i in range(len(video)):
                frame = cv2.resize(video[i], (self.width, self.height))

                # Fix for gray scale
                if len(frame.shape) == 2:
                    frame = np.expand_dims(frame, -1)
                resized_video[i] = frame
            return {"data": resized_video, "label": label}

    class Normalize(object):
        """ Normalize the image so that its values are in [0, 1] """

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            return {"data": video/255.0, "label": label}

    class VerticalFlip(object):
        """ Randomly flips the video around the x-axis """

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            if random.random() > 0.5:
                video = [np.flipud(frame) for frame in video]
            return {"data": video, "label": label}

    class HorizontalFlip(object):
        """ Randomly flips the video around the y-axis """

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            if random.random() > 0.5:
                video = [np.fliplr(frame) for frame in video]
            return {"data": video, "label": label}

    class Rotate180(object):
        """ Randomly rotate the video by 180 degrees """

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            if random.random() > 0.5:
                video = [np.rot90(frame, 2) for frame in video]
            return {"data": video, "label": label}

    class ReverseTime(object):
        """ Randomly plays a video backward """

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            video = np.flip(video, axis=0).copy()   # .copy() is to make PyTorch happy
            return {"data": video, "label": label}

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            video = video.transpose((0, 3, 1, 2))
            return {"data": torch.from_numpy(video),
                    "label": torch.from_numpy(np.asarray(label))}

    class Noise(object):
        """ Add random noise to the image """

        def __call__(self, sample):
            video, label = sample["data"], sample["label"]
            noise_offset = (torch.rand(video[0].shape)-0.5)*0.05
            noise_scale = (torch.rand(video[0].shape) * 0.2) + 0.9

            video = video * noise_scale + noise_offset
            video = torch.clamp(video, 0, 1)

            return {"data": video, "label": label}
