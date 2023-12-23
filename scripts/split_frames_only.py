import cv2
import os
import argparse
import concurrent.futures
import logging


class VideoObject:
    def __init__(self, root, file):
        self.root = root
        self.file = file
        self.video = os.path.join(root, file)

    def frame_split(self, fps, dir="frames"):
        cap = cv2.VideoCapture(self.video)
        fps_original = int(cap.get(cv2.CAP_PROP_FPS))
        downsample = fps_original // fps
        out_dir = os.path.join(self.root, dir)
        os.makedirs(out_dir, exist_ok=True)

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % downsample == 0:
                outpath = os.path.join(
                    out_dir, f'{self.file.split(".")[0]}_frame{idx}.jpg'
                )
                cv2.imwrite(outpath, frame)
            idx += 1

        cap.release()
        cv2.destroyAllWindows()


def process_video(root, file, fps):
    video = VideoObject(root, file)
    video.frame_split(fps)
    logging.info(f"Frames split for video: {file}")


def main():
    parser = argparse.ArgumentParser(description="Split MP4 videos into frames")
    parser.add_argument("dirpath", help="Folder containing MP4 videos")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for root, dirs, files in os.walk(args.dirpath):
            for file in files:
                if file.endswith(".MP4"):
                    executor.submit(process_video, root, file, 0.2)


if __name__ == "__main__":
    main()
