import os
import streamlit as st

st.set_page_config(
    page_title="EgoViz Label",
    page_icon="🎥",
    layout="wide",
    # initial_sidebar_state="collapsed",
)

st.title("🎥 EgoViz Labeller")
st.subheader("A tool for labelling ADLs in egocentric videos.")

dir = st.text_input("Enter absolute path to videos directory...")
st.write("Current directory: ", dir)

# get all videos in directory, including subdirectories
videos = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".mp4") or file.endswith(".MP4"):
            videos.append(os.path.join(root, file))

# get paths for all videos in directory
video_paths = []
for video in videos:
    video_paths.append(os.path.join(dir, video))

st.write("Videos found: ", video_paths)

# load one video at a time and display a "next" and "previous" button to cycle through videos

col1, col2 = st.columns([1, 3])

with col1:
    current_video_index = 0
    st.write("Current video: ", video_paths[current_video_index])

    current_video_index = st.selectbox()

with col2:
    st.video(video_paths[current_video_index])
