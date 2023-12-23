import streamlit as st

from utils import (
    get_subclip_paths,
    event_listener,
    save_labels,
    delete_session_state,
    increment_idx,
    decrement_idx,
)


def main():
    st.set_page_config(
        page_title="EgoViz Label",
        page_icon="🎥",
        layout="wide",
    )

    st.title("🎥 EgoViz Labeller")
    st.subheader("A tool for labelling ADLs in egocentric videos.")

    dir_path = st.text_input(
        "Enter absolute path to videos directory...",
        on_change=delete_session_state,
    )

    video_paths = get_subclip_paths(dir_path)

    with st.expander("Videos Found"):
        st.write(video_paths)

    # Initialize session state variables

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    if "videos" not in st.session_state:
        st.session_state.videos = video_paths

    if "num_videos" not in st.session_state:
        st.session_state.num_videos = len(video_paths)

    if "labels" not in st.session_state:
        st.session_state.labels = {}

    col1, col2 = st.columns([1, 3])

    # Render ADL classification form
    with col1:
        btn1, btn2 = st.columns(2)
        with btn1:
            st.button("Previous", on_click=decrement_idx, use_container_width=True)
        with btn2:
            st.button("Next", on_click=increment_idx, use_container_width=True)

        current_video = str(st.session_state.videos[st.session_state.idx])

        with st.form("adl_form", clear_on_submit=True):
            adl = st.radio(
                label="ADL Classification",
                options=[
                    "communication-mgmt",
                    "functional-mobility",
                    "grooming-health-mgmt",
                    "home-management",
                    "leisure-other",
                    "meal-prep-cleanup",
                    "self-feeding",
                ],
                index=None,
            )
            submit = st.form_submit_button(label="Submit")

            if submit:
                st.session_state.labels[current_video] = adl
                st.success(f"Label saved as {adl}!")

        st.write(
            "Completed Videos: ",
            len(st.session_state.labels),
            "/",
            st.session_state.num_videos,
        )

    # Render video player
    with col2:
        st.video(current_video)

    # Render save button
    filename = st.text_input("Enter filename to save labels as...", value="labels")
    st.button(
        "Save Labels",
        on_click=save_labels,
        args=(st.session_state.labels, filename),
        use_container_width=True,
    )
    st.write("Labels:", st.session_state.labels)


if __name__ == "__main__":
    main()
    event_listener()
