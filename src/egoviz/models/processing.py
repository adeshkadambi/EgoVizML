import pickle


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data
