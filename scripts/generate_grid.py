import cv2
from PIL import Image


def extract_frames(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [
        int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)
    ]
    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def resize_frame(frame, target_size):
    height, width, _ = frame.shape
    target_width, target_height = target_size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # Landscape orientation
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Portrait orientation
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def create_grid(frames, grid_size=(3, 2), target_size=(672, 672)):
    grid_width = grid_size[1]
    grid_height = grid_size[0]
    frame_height, frame_width, _ = frames[0].shape

    resized_frames = [resize_frame(frame, target_size) for frame in frames]

    grid_image = Image.new(
        "RGB", (target_size[0] * grid_width, target_size[1] * grid_height)
    )

    for i, frame in enumerate(resized_frames):
        x_offset = (i % grid_width) * target_size[0]
        y_offset = (i // grid_width) * target_size[1]
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        grid_image.paste(pil_frame, (x_offset, y_offset))

    # Resize grid image to fit within 672x672
    grid_image = grid_image.resize((target_size[0], target_size[1]))

    return grid_image


def main(video_path):
    frames = extract_frames(video_path)
    grid_image = create_grid(frames)
    output_file = (
        video_path.split(".")[0] + ".jpg"
    )  # Assumes input video has an extension
    grid_image.save(output_file)
    print(f"Grid image saved as: {output_file}")


if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    main(video_path)
