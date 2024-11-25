# EgoVizML

This repo contains all code pertaining to my thesis titled "Using wearable technology to inform clinical decision-making in outpatient neurorehabilitation". The overarching goal of this thesis is to identify the factors that affect clinical decision making in outpatient rehabilitation care and determine if wearable technology can deliver actionable information about at-home upper limb performance to better guide care delivery.

More specifically, I hope to:

- **(Aim 1)** evaluate the perceived usefulness of delivering hand function metrics extracted from wearable technologies to clinicians in outpatient care,
- **(Aim 2)** provide at-home hand-use context through the detection of object interactions and activities of daily living (ADLs),
- **(Aim 3)** explore the decision-making processes and challenges in the delivery of upper-limb neurorehabilitation to determine clinical decision support needs.

# Activity Recognition Pipeline Documentation

## Overview

This pipeline detects daily living activities by analyzing object interactions in egocentric videos. It uses:

1. DETIC object detector
2. Hand-object interaction detector
3. Logistic Regression classifier

## Pipeline Steps

### 1. Object Classification

`process_detic_data.py` processes raw object detections:

- Maps 1000+ DETIC classes to 29 functional categories using CSV mapping
- Filters out human detections
- Saves processed predictions as pickle files

### 2. Active Object Detection

`process_all_preds.py` identifies objects being interacted with:

- Combines DETIC object boxes with hand interaction boxes
- Labels objects as "active" if IoU with hand box > 0.75
- Organizes by activity type and video
- Saves combined predictions as `all_preds.pkl`

### 3. Feature Generation

`processing.py` creates features for classification:

- `generate_df_from_preds()`: Converts predictions to DataFrame
- `generate_binary_presence_df()`: Creates binary features (object present/not present)
- `generate_counts_df()`: Creates count-based features
- `row_wise_min_max_scaling()`: Normalizes features per video

### 4. Model Training and Evaluation

`evaluation.py` handles model training/testing:

- Uses leave-one-subject-out cross-validation
- Best configuration: Binary + Active features with Logistic Regression
- Performance: 0.78 mean F1-score across subjects

## Usage

```python
# 1. Generate active object labels
python process_all_preds.py /path/to/data --active_iou 0.75

"""
The pipeline expects object detection and hand-object interaction predictions organized in this structure:

root_directory/
├── communication-management/
│   ├── detic/
│   │   ├── video1_frame1.pkl
│   │   └── ...
│   └── shan/
│       ├── video1_frame1.pkl
│       └── ...
├── functional-mobility/
│   ├── detic/
│   └── shan/
└── ...other activity folders...

It will output a dictionary.
"""

# 2. Generate features for inference
from egoviz.models import processing
from egoviz.models import inference

new_data = processing.generate_df_from_preds(preds_dict_from_step1)
new_data = processing.generate_binary_presence_df(new_data)
scaled_data = processing.row_wise_min_max_scaling(new_data)

# 3. Load model
model = inference.load_production_model("models/binary_active_logreg.joblib")

# 4. Make predictions
predictions = inference.predict(scaled_data, model)

# 5. Access results
activities = predictions.select('predicted_label')
probabilities = predictions.select(pl.col('^prob_.*$'))
```

## Data Formats

### Raw DETIC Predictions

```python
{
    'boxes': list[list[int]],  # Bounding boxes [x1,y1,x2,y2]
    'scores': list[float],     # Detection confidence
    'classes': list[int],      # Original DETIC class IDs
    'metadata': list[str]      # Original class names
}
```

### Processed Predictions

```python
{
    # Original fields +
    'remapped_metadata': list[str],  # Functional category names
    'remapped_classes': list[int],   # Functional category IDs
    'active_objects': list[bool]     # Hand interaction flags
}
```

### Final Features

Binary + Active configuration creates columns:

- `{object_class}`: Binary presence (0/1)
- `active_{object_class}`: Binary interaction (0/1)
- All features scaled per video using min-max scaling

## Dependencies

- PyTorch (torchvision)
- scikit-learn
- pandas
- polars
- numpy

# Project Setup

1. Clone the repository

2. Install dependencies

```bash
pip install poetry
poetry install
```

3. Activate the virtual environment

```bash
poetry shell
```

# Dependency Management

You can add and remove dependencies using the following:

```bash
poetry add <package>
poetry remove <package>
```

To update the egoviz package:

```bash
pip uninstall egoviz
poetry install
```

# ADL Detection Classes

Defined by the American Occupational Therapy Association (AOTA) in the Occupational Therapy Practice Framework: Domain and Process (2020).

## Feeding

_Setting up, arranging, and bringing food [or fluid] from the plate or cup to the mouth; sometimes called self-feeding._

## Functional Mobility

_Moving from one position or place to another (during performance of everyday activities), such as in-bed mobility, wheelchair mobility, and transfers (e.g., wheelchair, bed, car, shower, tub, toilet, chair, floor). Includes functional ambulation and transportation of objects._

## Grooming / Health Management

_Obtaining and using supplies; removing body hair (e.g., using razor, tweezers, lotion); applying and removing cosmetics; washing, drying, combing, styling, brushing, and trimming hair; caring for nails (hands and feet); caring for skin, ears, eyes, and nose; applying deodorant; cleaning mouth; brushing and flossing teeth; and removing, cleaning, and reinserting dental orthotics and prosthetics. Developing, managing, and maintaining routines for health and wellness promotion, such as physical fitness, nutrition, decreased health risk behaviors, and medication routines._

## Communication Management

_Sending, receiving, and interpreting information using a variety of systems and equipment, including writing tools, telephones (cell phones or smartphones), keyboards, audiovisual recorders, computers or tablets, communication boards, call lights, emergency systems, Braille writers, telecommunication devices for deaf people, augmentative communication systems, and personal digital assistants._

## Home Establishment and Management

_Obtaining and maintaining personal and household possessions and environment (e.g., home, yard, garden, appliances, vehicles), including maintaining and repairing personal possessions (e .g ., clothing, household items) and knowing how to seek help or whom to contact._

## Meal Preparation and Cleanup

_Planning, preparing, and serving well-balanced, nutritious meals and cleaning up food and utensils after meals._

## Leisure and Other

_Nonobligatory activity that is intrinsically motivated and engaged in during discretionary time, that is, time not committed to obligatory occupations or any of the aforementioned ADLs or iADLs._
