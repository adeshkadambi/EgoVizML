"""
This script splits MP4 videos into subclips and frames.

Example usage:
python video_to_subclips_and_frames.py /path/to/your/videos --subclip_length 60 --fps 10 --frame_fps 2

This will create a folder called "subclips" in the same directory as the videos, 
and a folder for each subclip containing frames from that subclip.
"""

import argparse

from egoviz.cdss_utils.video_processing import process_videos_in_folder


def main():
    parser = argparse.ArgumentParser(
        description="Split MP4 videos into subclips and frames."
    )
    parser.add_argument("dirpath", help="Folder containing MP4 videos")
    parser.add_argument(
        "--subclip_length",
        type=int,
        default=60,
        help="Subclip length in seconds (default: 60)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for subclips (default: 10)",
    )
    parser.add_argument(
        "--frame_fps",
        type=int,
        default=2,
        help="Frames per second for frames (default: 2)",
    )
    parser.add_argument(
        "--frames_only",
        action="store_true",
        help="Split videos into frames only, skipping subclip creation",
    )
    args = parser.parse_args()

    process_videos_in_folder(
        args.dirpath, args.subclip_length, args.fps, args.frame_fps, args.frames_only
    )


if __name__ == "__main__":
    main()
