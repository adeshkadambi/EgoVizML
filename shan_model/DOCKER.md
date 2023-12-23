### To build the comtainer:
```bash
docker build -t shan-container . -f Dockerfile
```

### To run the container:
```bash
docker run -it --rm --gpus all -v $(pwd):/app shan-container
```

### When inside the container:
```bash
apt-get update
apt-get install python3-opencv -y
```

### To run the model:
```bash
python3.8 run_shan.py --image_dir images
```

Note: Make sure there is a folder named `frames` in the image_dir path
with the frames of the video that you want to run the model on. The model will
create a folder named `shan` in the image_dir path with the output frames.