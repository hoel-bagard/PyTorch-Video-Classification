import argparse

import cv2


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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while True:
            cv2.imshow("frame", gray)
            if cv2.waitKey(10) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
