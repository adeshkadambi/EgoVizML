import os
import re
import json
import tqdm
import pickle

import egoviz.cdss_utils.metrics as egomet

# Read the JSON file
with open("./data/dashboard_video_keys.json") as f:
    data = json.load(f)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


# for each entry in video_key, create a dictionary to hold metrics and frames for each day
metrics: dict[str, dict[str, float]] = {}
frames: dict[str, list] = {}

key = input("Enter the name of the video key: ")
video_key = data[key]

for date, videos in video_key.items():
    metrics[date] = {}
    frames[date] = []

# root directory from user input
root = input("Enter the path to root dir (i.e., subclips_shan): ")

# pre-sort the list of subdirectories
subdirs = sorted_alphanumeric(os.listdir(root))

# for each subdirectory in root, get the date by checking if the name is in the video_key dict
for subdir in tqdm.tqdm(subdirs):
    # if any of the lists in video_key are a substring of subdir, get the date
    for date, videos in video_key.items():
        if any(video in subdir for video in videos):
            # get the date
            video_date = date

    # load all .pkl files in the subdirectory
    for file in sorted_alphanumeric(os.listdir(os.path.join(root, subdir))):
        if file.endswith(".pkl"):
            # load the file and append to frames
            try:
                with open(os.path.join(root, subdir, file), "rb") as p:
                    frames[video_date].append(pickle.load(p))
            except EOFError:
                print(f"{os.path.join(root, subdir, file)} is corrupt")  # type: ignore

# for each date in frames, compute metrics and store in metrics
for date, frames_list in frames.items():
    metrics[date]["interaction_percentage"] = egomet.interaction_percentage(frames_list)
    metrics[date]["interactions_per_hour"] = egomet.interactions_per_hour(frames_list)
    metrics[date]["average_interaction_duration"] = egomet.average_interaction_duration(
        frames_list
    )

# save metrics to json file and log to console
print(metrics)

write_path = input("Enter the path to save the metrics: ")
# convert write path to path object
write_path = os.path.join(write_path, f"{key}_metrics.json")

with open(write_path, "w", encoding="utf-8") as f:  # specify encoding as utf-8
    json.dump(metrics, f)
