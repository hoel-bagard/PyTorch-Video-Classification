from argparse import ArgumentParser
from pathlib import Path
import shutil

import cv2


def main():
    parser = ArgumentParser("Script to (optionally) crop and resize the videos, then save them as images")
    parser.add_argument("data_path", type=Path, help='Path to directory to read for data')
    parser.add_argument("output_path", type=Path, help='Path to output dataset')
    parser.add_argument("--crop", nargs=4, type=int, help='Input frame crop')
    parser.add_argument("--resize-ratio", "--r", type=float, help='Ratio of resize')
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    default_fps = None
    video_paths = list(data_path.glob("**/*.avi"))
    nb_videos = len(video_paths)
    for video_index, video_path in enumerate(video_paths, start=1):
        msg = f"Processing file {video_path},   ({video_index}/{nb_videos})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        if default_fps is None:
            default_fps = fps
        if fps != default_fps:
            print(f'Warning : {video_path} FPS is different : {fps} instead of {default_fps}')
            continue
        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
            if args.crop:
                frame = frame[args.crop[0]:args.crop[1], args.crop[2]:args.crop[3], :]
            if args.resize_ratio:
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * args.resize_ratio), int(frame.shape[0] * args.resize_ratio)))

            # Saves image of the current frame in jpg file
            img_output_path = output_path / str(video_path)[:-4] / (str(frame_count).zfill(3) + ".jpg")
            img_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_count += 1

    print("\nFinished.")


if __name__ == "__main__":
    main()
