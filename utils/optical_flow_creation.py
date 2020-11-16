import argparse
import os
import glob

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser("Generate optical flow videos for a dataset")
    parser.add_argument('data_path', help='Path to the dataset')
    args = parser.parse_args()

    file_types = ("*.avi", "*.mp4")
    video_paths = []
    [video_paths.extend(glob.glob(os.path.join(args.data_path, "**", ext), recursive=True)) for ext in file_types]
    for i, video_path in enumerate(video_paths):
        # Open video and get its properties
        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = frame.shape[0:2]
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        # Define the codec and create VideoWriter object
        video_writter = cv2.VideoWriter(os.path.splitext(video_path)[0]+"_optical_flow.mp4",
                                        # cv2.VideoWriter_fourcc(*"X264"),
                                        cv2.VideoWriter_fourcc(*"avc1"),
                                        cap.get(cv2.CAP_PROP_FPS),
                                        (width, height),
                                        isColor=True)

        frame_nb = 0
        while(cap.isOpened()):
            print(f"Processing video {video_path} (frame {frame_nb}/{video_length})   ", flush=True, end="\r")
            ret, frame = cap.read()
            if ret:
                next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                previous_frame = next_frame
                frame_nb += 1
                video_writter.write(bgr)

                # while True:
                #     cv2.imshow("Frame", bgr)
                #     if cv2.waitKey(10) == ord("q"):
                #         break

            else:
                break

        # Release everything
        cap.release()
        video_writter.release()
        print("\nFinished processing video {video_path}")


if __name__ == "__main__":
    main()
