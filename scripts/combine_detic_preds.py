"""
This script combines the processed and remapped predictions of the detic model into one
file. The predictions are saved in the following format:

all_preds[{adl}_{video}_frame{#}] = {preds for frame #}
"""

import argparse
import os
import pickle
import logging

from tqdm import tqdm
from egoviz.models.processing import load_pickle


def combine_detic_preds(dirpath: str):
    all_preds = {}
    adls = [
        "communication-management",
        "functional-mobility",
        "grooming-health-management",
        "home-management",
        "leisure-other-activities",
        "meal-preparation-cleanup",
        "self-feeding",
        # "test_folder"
    ]

    for adl in adls:
        logging.info(f"Processing {adl}...")

        adl_dirpath = os.path.join(dirpath, adl, "detic")

        if os.path.exists(adl_dirpath):
            pkl_files = [f for f in os.listdir(adl_dirpath) if f.endswith(".pkl")]

            for file in tqdm(pkl_files):
                filepath = os.path.join(adl_dirpath, file)
                preds = load_pickle(filepath)

                # get video name
                video = file.split("_")[0] + "_" + file.split("_")[1]

                # add to all_preds
                all_preds[f"{adl}_{video}"] = preds

    # save all_preds
    savepath = os.path.join(dirpath, "all_preds.pkl")
    with open(savepath, "wb") as f:
        pickle.dump(all_preds, f)

    return 1


def main():
    parser = argparse.ArgumentParser(description="Combine processed detic preds")
    parser.add_argument("dirpath", help="Folder containing pkl files")

    args = parser.parse_args()

    print(f"Combining processed detic preds in {args.dirpath}...")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    combine_detic_preds(args.dirpath)


if __name__ == "__main__":
    main()
