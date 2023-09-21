# EgoVizML

This repo contains all code pertaining to my thesis titled "Using wearable technology to inform clinical decision-making in outpatient neurorehabilitation". The overarching goal of this thesis is to identify the factors that affect clinical decision making in outpatient rehabilitation care and determine if wearable technology can deliver actionable information about at-home upper limb performance to better guide care delivery.

More specifically, I hope to: 
- **(Aim 1)** evaluate the perceived usefulness of delivering hand function metrics extracted from wearable technologies to clinicians in outpatient care,
- **(Aim 2)** provide at-home hand-use context through the detection of object interactions and activities of daily living (ADLs),
- **(Aim 3)** explore the decision-making processes and challenges in the delivery of upper-limb neurorehabilitation to determine clinical decision support needs.

## Project Organization

```
EgoVizML (root)
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
```
