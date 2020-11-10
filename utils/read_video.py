import argparse

import cv2
import numpy as np


def crop(frame, left: int = 0, right: int = 1, top: int = 0, bottom: int = 1):
    frame = frame[top:-bottom, left:-right]
    return frame


def main():
    parser = argparse.ArgumentParser("Displays a video frame by frame")
    parser.add_argument('data_path', help='Path to the video')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.data_path)
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 130)

    while(cap.isOpened()):
        ret, frame = cap.read()

        # Crop (using the same function as in train)
        # frame = crop(frame, top=900, bottom=400)    # Used for September 2020 data
        # frame = crop(frame, top=200, bottom=1100)    # Used for October 2020 data, needs to be adjusted on a per folder basis
        frame = np.asarray(frame, dtype=np.uint8)

        # To grayscale
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while True:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(10) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
