import os
import json
import tempfile
from typing import (
    Dict,
    List
)

import cv2
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch

from config.model_config import ModelConfig


def dali_n_to_n_file_list(data_path: str, label_map: Dict, limit: int = None) -> str:
    """
    Builds a file list for the DALI VideoReader (for the N to N case).
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the videos inside.
                   It should also contain a label.json file with the labels (file paths and time stamps)
        label_map: dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
    Return:
        filelist (strings with format "file_path cls start end\n")
    """
    # Read label file
    label_file_path = os.path.join(data_path, "labels.json")
    assert os.path.isfile(label_file_path), "Label file is missing"

    for key, value in label_map.items():
        if(value == "not_visible"):
            not_visible_cls = key
            break
    assert "not_visible_cls" in locals(), "There should be a 'not_visible' class"

    with open(label_file_path) as json_file:
        json_labels = json.load(json_file)
        labels = json_labels["entries"]

    nb_labels = len(labels)

    label_list = ""     # Instruction for the DALI VideoReader

    for i, label in enumerate(labels, start=1):
        video_path = os.path.join(data_path, label["file_path"])
        msg = f"Loading data {video_path}    ({i}/{nb_labels})"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end="\r")

        assert os.path.isfile(video_path), f"Video {video_path} is missing"
        cap = cv2.VideoCapture(video_path)  # Use ffprobe instead ?
        video_length: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        for key in range((len(label_map))):
            if label_map[key] in label["file_path"]:
                video_cls = key
                break
        assert video_cls, f"There is no class corresponding to {label['file_path']}"

        # The first toggle in the label corresponds to when the object becomes visible
        # It can be visible from the start. Also, there can be no timestamp if it is never visible
        if len(label["time_stamps"]) != 0 and label["time_stamps"][0] != 0:
            label["time_stamps"].insert(0, 0)
            visibility_status = False
        else:
            visibility_status = True

        for i in range(len(label["time_stamps"])):
            video_extract_cls = video_cls if visibility_status else not_visible_cls
            if i != len(label["time_stamps"])-1:
                beginning, end = label["time_stamps"][i], label["time_stamps"][i+1]
            else:
                beginning, end = label["time_stamps"][i], video_length

            # test_label.mp4 1 0 100 is interpreted as applying label 1 from frame number 0 to 100(excluding).
            label_list += f"{video_path} {video_extract_cls} {beginning} {end}\n"
            visibility_status = not visibility_status
        if limit and i == limit:
            break

    return label_list


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, file_list, crop_size, mode: str = "Train"):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.reader = ops.VideoReader(device="gpu", file_list=file_list, sequence_length=sequence_length,
                                      normalized=False, random_shuffle=(mode == "Train"),
                                      initial_fill=4*ModelConfig.BATCH_SIZE,  # Size of the buffer for shuffling.
                                      image_type=types.GRAY if ModelConfig.USE_GRAY_SCALE else types.RGB,
                                      dtype=types.UINT8,
                                      file_list_frame_num=True, enable_frame_num=True, enable_timestamps=True,
                                      pad_last_batch=False)  # If True, pads the shard by repeating the last sample.

        # self.crop = ops.Crop(device="gpu", crop=crop_size, dtype=types.FLOAT)
        # self.uniform = ops.Uniform(range=(0.0, 1.0))
        # self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])

    def define_graph(self):
        input = self.reader(name="Reader")
        # cropped = self.crop(input, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        # output = self.transpose(cropped)
        return input


class DALILoader():
    def __init__(self, data_path: str, label_map: Dict, limit: int = None,
                 crop_size: List[int, int] = [256, 256]):
        """
        Args:
            data_path: Path to the root folder of the dataset.
                       This folder is expected to contain subfolders for each class, with the videos inside.
                       It should also contain a label.json file with the labels (file paths and time stamps)
            label_map: dictionarry mapping an int to a class
            limit (int, optional): If given then the number of elements for each class in the dataset
                                   will be capped to this number
        """

        # Make list of files and associated labels
        label_list = dali_n_to_n_file_list(data_path, label_map, limit=limit)
        tf = tempfile.NamedTemporaryFile()
        tf.write(str.encode(label_list))
        tf.flush()
        file_list = tf.name
        self.pipeline = VideoReaderPipeline(batch_size=ModelConfig.BATCH_SIZE,
                                            sequence_length=ModelConfig.VIDEO_SIZE,
                                            num_threads=2,
                                            device_id=0,
                                            file_list=file_list,
                                            crop_size=crop_size)
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data"],
                                                         reader_name="Reader",
                                                         fill_last_batch=False,
                                                         last_batch_padded=True,
                                                         auto_reset=True)

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()
