import pickle
import pandas as pd


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def generate_dataframe_from_preds(preds: dict[dict]):
    # check if each key in preds has a key "active"
    df = pd.DataFrame(columns=["video", "frame", "classes", "active", "adl"])

    for idx, dets in preds.items():
        adl = idx.split("_", 1)[0]
        video = idx.split("_")[1]
        frame = idx.split("_")[2]
        classes = dets["remapped_metadata"]
        active = dets["active_objects"]

        row = {
            "video": video,
            "frame": frame,
            "classes": classes,
            "adl": adl,
            "active": active,
        }

        df.loc[len(df)] = row
        return df
