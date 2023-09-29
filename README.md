# EgoVizML

This repo contains all code pertaining to my thesis titled "Using wearable technology to inform clinical decision-making in outpatient neurorehabilitation". The overarching goal of this thesis is to identify the factors that affect clinical decision making in outpatient rehabilitation care and determine if wearable technology can deliver actionable information about at-home upper limb performance to better guide care delivery.

More specifically, I hope to:

- **(Aim 1)** evaluate the perceived usefulness of delivering hand function metrics extracted from wearable technologies to clinicians in outpatient care,
- **(Aim 2)** provide at-home hand-use context through the detection of object interactions and activities of daily living (ADLs),
- **(Aim 3)** explore the decision-making processes and challenges in the delivery of upper-limb neurorehabilitation to determine clinical decision support needs.

## Project Organization

```
EgoVizML (root of repository)
|-- egoviz (package)
  |-- cdss_utils/
  > processing images and videos to prep them for models or the egoviz dashboard.
  |  |-- video_processing.py
  |  |-- dashboard_metrics.py

  |-- models/
  > scripts for running various models on frames or feature vectors.
  |  |-- hand_object_detectors/
  |  |  |-- shan.py
  |  |  |-- egohos.py
  |  |-- object_detection/
  |  |  |-- detic.py
  |  |  |-- unidet.py
  |  |-- adl_detection/
  |  |  |-- logistic_regression.py
  |  |  |  ...

  |-- bag_of_objects/
  > scripts for generating feature vectors from frames.
  |  |-- frequency_counts.py
  |  |-- tf-idf.py
  |  |-- binary_presence.py
  |  |-- word_embedding.py
  |  |-- n-grams.py

|-- data/
> contains all data used in the project.

|-- scripts/
> contains scripts for running models and generating feature vectors from data.
```

### Project Setup

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

### Dependency Management

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

## ADL Detection Classes

Defined by the American Occupational Therapy Association (AOTA) in the Occupational Therapy Practice Framework: Domain and Process (2020).

#### Feeding

_Setting up, arranging, and bringing food [or fluid] from the plate or cup to the mouth; sometimes called self-feeding._

#### Functional Mobility

_Moving from one position or place to another (during performance of everyday activities), such as in-bed mobility, wheelchair mobility, and transfers (e.g., wheelchair, bed, car, shower, tub, toilet, chair, floor). Includes functional ambulation and transportation of objects._

#### Grooming / Health Management

_Obtaining and using supplies; removing body hair (e.g., using razor, tweezers, lotion); applying and removing cosmetics; washing, drying, combing, styling, brushing, and trimming hair; caring for nails (hands and feet); caring for skin, ears, eyes, and nose; applying deodorant; cleaning mouth; brushing and flossing teeth; and removing, cleaning, and reinserting dental orthotics and prosthetics. Developing, managing, and maintaining routines for health and wellness promotion, such as physical fitness, nutrition, decreased health risk behaviors, and medication routines._

#### Communication Management

_Sending, receiving, and interpreting information using a variety of systems and equipment, including writing tools, telephones (cell phones or smartphones), keyboards, audiovisual recorders, computers or tablets, communication boards, call lights, emergency systems, Braille writers, telecommunication devices for deaf people, augmentative communication systems, and personal digital assistants._

#### Home Establishment and Management

_Obtaining and maintaining personal and household possessions and environment (e.g., home, yard, garden, appliances, vehicles), including maintaining and repairing personal possessions (e .g ., clothing, household items) and knowing how to seek help or whom to contact._

#### Meal Preparation and Cleanup

_Planning, preparing, and serving well-balanced, nutritious meals and cleaning up food and utensils after meals._

#### Leisure and Other

_Nonobligatory activity that is intrinsically motivated and engaged in during discretionary time, that is, time not committed to obligatory occupations or any of the aforementioned ADLs or iADLs._
