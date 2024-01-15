import os
import argparse

from icecream import ic


def filesearch(root_dir):
    """
    Search through root_dir and return:
    - a list of paths to jpg files
    - a list of paths to jpg files with root dir modified from "subclips" to "subclips_shan"
    """

    img_list = []

    for root, dirs, _ in os.walk(root_dir):
        for directory in dirs:
            if any(
                fname.endswith(".jpg")
                for fname in os.listdir(os.path.join(root, directory))
            ):
                frames_path = os.path.join(root, directory)
                shan_path = frames_path.replace("subclips", "subclips_shan")
                for file in os.listdir(frames_path):
                    if file.endswith(".jpg"):
                        img_path = os.path.join(frames_path, file)
                        save_img_path = os.path.join(shan_path, file[:-4] + ".png")
                        save_pkl_path = os.path.join(shan_path, file[:-4] + "_shan.pkl")
                        img_list.append((img_path, save_img_path, save_pkl_path))
    return img_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search through root_dir and return a list of paths to jpg files"
    )
    parser.add_argument("-r", "--root_dir", type=str, help="Root directory to search")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    files = filesearch(args.root_dir)
    ic(len(files))
    ic(files[0])
