def default_loader(data_path: str, label_map: Dict, load_videos: bool = False) -> np.ndarray:
    """
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the videos inside.
        label_map: dictionarry mapping an int to a class
        load_videos: If true then this function returns the videos instead of their paths
    Return:
        numpy array containing the paths/videos and the associated label
    """
    data = []
    for key in range(len(label_map)):

        for video_path in glob.glob(os.path.join(data_path, self.label_map[key], "*.avi")):
            print(f"Loading data {video_path}   ", end="\r")
            labels.append([video_path, key])        


        file_types = ("*.avi", "*.mp4")
        pathname = os.path.join(data_path, label_map[key], "**")
        video_paths = []
        [video_paths.extend(glob.glob(os.path.join(pathname, ext), recursive=True)) for ext in file_types]
        for video_path in video_paths:
            print(f"Loading data {video_path}   ", end="\r")
            if load_videos:
                cap = cv2.VideoCapture(video_path)
                video = []
                while(cap.isOpened()):
                    frame_ok, frame = cap.read()
                    if frame_ok:
                        if ModelConfig.USE_GRAY_SCALE:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = np.expand_dims(frame, -1)  # To keep a channel dimension (gray scale)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video.append(video)
                cap.release()
                data.append([video, key])
            else:
                data.append([video_path, key])
    data = np.asarray(data)
    return data