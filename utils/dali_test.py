import argparse
import os
import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


sequence_length = 8
initial_prefetch_size = 16


class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data, shuffle):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)
        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=initial_prefetch_size)

    def define_graph(self):
        output = self.input(name="Reader")
        return output


def main():
    parser = argparse.ArgumentParser("Displays a video frame by frame")
    parser.add_argument('data_path', help='Path to the video')
    args = parser.parse_args()

    video_directory = args.data_path
    video_files = [video_directory + '/' + f for f in os.listdir(video_directory)]

    batch_size = 2
    shuffle = True
    n_iter = 6

    pipe = VideoPipe(batch_size=batch_size, num_threads=2, device_id=0, data=video_files, shuffle=shuffle)
    pipe.build()
    for i in range(n_iter):
        pipe_out = pipe.run()
        sequences_out = pipe_out[0].as_cpu().as_array()
        print(sequences_out.shape)


if __name__ == "__main__":
    main()
