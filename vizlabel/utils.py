import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def increment_idx():
    if st.session_state.idx < st.session_state.num_videos - 1:
        st.session_state.idx += 1
    else:
        st.info("No more videos to label!")


def decrement_idx():
    if st.session_state.idx > 0:
        st.session_state.idx -= 1
    else:
        st.info("Already at first video!")


def delete_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]


def get_subclip_paths(dir_path: str) -> [Path]:
    """Returns a list of paths to all subclips in a directory."""
    dir_path = Path(dir_path)
    mp4_paths = []

    # Iterate through the directory and its subdirectories
    for item in dir_path.glob("**/*"):
        if item.is_dir() and item.name == "subclips":
            for mp4_file in item.glob("*.mp4"):
                mp4_paths.append(mp4_file)

    return mp4_paths


def event_listener():
    return components.html(
        """
        <script>
        const doc = window.parent.document;
        buttons = Array.from(doc.querySelectorAll('button'));
        const left_button = buttons.find(el => el.innerText === 'Previous');
        const right_button = buttons.find(el => el.innerText === 'Next');
        doc.addEventListener('keydown', function(e) {
            switch (e.keyCode) {
                case 37: // (37 = left arrow)
                    left_button.click();
                    break;
                case 39: // (39 = right arrow)
                    right_button.click();
                    break;
            }
        });
        </script>
        """,
        height=0,
        width=0,
    )


def save_labels(state_dict: dict, filename: str):
    with open(f"vizlabel/labels/{filename}.json", "w") as f:
        json.dump(state_dict, f)
