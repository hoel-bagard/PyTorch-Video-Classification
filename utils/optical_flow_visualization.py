import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser("Visualize optical flow on a video")
    parser.add_argument('video_path', help='Path to the video')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    _, frame = cap.read()
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = frame.shape[0:2]
    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    flow_video = []
    frame_nb = 0
    while(cap.isOpened()):
        print(f"Processing frame  ({frame_nb}/{video_length})   ", flush=True, end="\r")
        ret, frame = cap.read()
        if ret:
            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            previous_frame = next_frame
            frame_nb += 1
            flow_video.append(bgr)
        else:
            break


    for frame in flow_video:
        while True:
            frame = cv2.resize(frame, (int(width/3), int(height/3)))
            cv2.imshow("Flow frame", frame)
            if cv2.waitKey(10) == ord("q"):
                break



if __name__ == "__main__":
    main()
