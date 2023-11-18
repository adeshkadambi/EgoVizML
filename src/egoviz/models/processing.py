from collections import Counter

import pickle
import pandas as pd


def load_pickle(filepath: str):
    """Loads a pickle file from a filepath."""

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def generate_df_from_preds(preds: dict[dict], save_path: str | None = None):
    """Generates a dataframe from the predictions of the model."""

    df = pd.DataFrame(columns=["video", "frame", "classes", "active", "adl"])

    for idx, dets in preds.items():
        adl = idx.split("_", 1)[0]
        video = idx.split("_")[1]
        frame = idx.split("_")[2]

        assert "remapped_metadata" in dets.keys(), "remapped_metadata not in dets"
        assert "active_objects" in dets.keys(), "active_objects not in dets"

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

    if save_path:
        df.to_pickle(save_path)

    return df


def generate_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a dataframe with counts of active/inactive objects per video."""

    def count_occurrences(classes, active):
        class_counts = Counter(classes)
        active_counts = Counter(
            {
                cls: sum([act and (cls == c) for act, c in zip(active, classes)])
                for cls in set(classes)
            }
        )
        return class_counts, active_counts

    df["class_counts"], df["active_counts"] = zip(
        *df.apply(lambda row: count_occurrences(row["classes"], row["active"]), axis=1)
    )

    # Create a new DataFrame from class_counts and active_counts
    counts_df = pd.DataFrame(
        df.apply(
            lambda row: {
                "adl": row["adl"],
                "video": row["video"],
                **{f"count_{key}": value for key, value in row["class_counts"].items()},
                **{
                    f"active_{key}": value
                    for key, value in row["active_counts"].items()
                },
            },
            axis=1,
        ).tolist()
    )

    # Group by video and sum the values for each video
    grouped_counts_df = counts_df.groupby("video").agg(
        {
            **{"adl": "first"},
            **{col: "sum" for col in counts_df.columns if col not in ["adl", "video"]},
        }
    )

    return grouped_counts_df.reset_index()
