import cv2
import os
import argparse
from moviepy.editor import VideoFileClip
from math import floor
import tqdm
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level (e.g., INFO, DEBUG)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def segments(duration, n=60):
    """
    Calculate start and end times for each subclip
    """
    return [(i, i + n) for i in range(0, duration, n)]


def save_subclips(video_path, root, n=60, fps=10, frame_fps=2):
    """
    Save subclips of a video
    """
    logger.info(f"Processing file: {video_path}")
    try:
        # Load video
        clip = VideoFileClip(video_path)

        # Get duration and start and end times
        duration = int(floor(clip.duration))
        times = segments(duration, n)

        # Create subclips and save
        subclip_folder = os.path.join(root, "subclips")
        os.makedirs(subclip_folder, exist_ok=True)  # Ensure the folder exists

        for idx, (start, end) in enumerate(times):
            subclip = clip.subclip(start, end)
            base_name = os.path.splitext(video_path)[0]
            clip_name = f"{base_name}--{idx + 1}.MP4"
            clip_path = os.path.join(subclip_folder, clip_name)

            logger.info(f"Saving subclip: {clip_path}")

            # Save videos with specified fps and no audio
            subclip.write_videofile(clip_path, fps=fps, audio=False)

            # Create a folder for frames
            frame_output_folder = os.path.splitext(clip_path)[0]
            os.makedirs(frame_output_folder, exist_ok=True)

            # Split subclip into frames and save
            frame_split(clip_path, frame_output_folder, fps=frame_fps)

        clip.close()
    except Exception as e:
        logger.error(f"Error: {e} for file: {video_path}")


def frame_split(video_path, output_folder, fps):
    """
    Split video into frames and save in output_folder
    """
    logger.info(f"Splitting frames for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps_original = int(cap.get(cv2.CAP_PROP_FPS))
    downsample = fps_original // fps
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % downsample == 0:
            outpath = os.path.join(output_folder, f"frame_{idx}.jpg")
            cv2.imwrite(outpath, frame)
        idx += 1

    cap.release()
    cv2.destroyAllWindows()


def process_videos_in_folder(dirpath, subclip_length=60, fps=10, frame_fps=2):
    """
    Split videos into subclips and frames in the same folder structure
    """
    # Iterate through all videos in dirpath
    for file in os.listdir(dirpath):
        if file.endswith(".MP4"):
            os.chdir(dirpath)
            save_subclips(file, dirpath, subclip_length, fps, frame_fps)
