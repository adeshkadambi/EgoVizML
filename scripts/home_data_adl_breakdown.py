import json
import matplotlib.pyplot as plt
import numpy as np

import os
from pathlib import PurePath

# load json file
filepath = os.path.join(
    PurePath(__file__).parents[1], "data", "patient_adl_breakdown.json"
)

with open(filepath) as f:
    data = json.load(f)

# create stacked bar chart for each patient in the data

# sort data by patient id
data = {k: data[k] for k in sorted(data)}

# create a list of patients
patients = list(data.keys())

# create a list of adls by getting all unique keys from all patients
adls = [
    "communication-management",
    "functional-mobility",
    "grooming-health-management",
    "home-management",
    "leisure-other-activities",
    "meal-preparation-cleanup",
    "self-feeding",
]

# create a list of values for each adl for each patient in the data
# use 0 for any adl that a patient does not have
values = []

for patient in patients:
    values.append([data[patient].get(adl, 0) for adl in adls])

values = np.array(values).T

# take the sum across each row to get total minutes recorded for each adl
total_minutes = np.sum(values, axis=1)

# create a list of labels for each adl
labels = [
    f"Communication Management ({total_minutes[0]} min total)",
    f"Functional Mobility ({total_minutes[1]} min total)",
    f"Grooming & Health Management ({total_minutes[2]} min total)",
    f"Home Management ({total_minutes[3]} min total)",
    f"Leisure & Other Activities ({total_minutes[4]} min total)",
    f"Meal Preparation / Cleanup ({total_minutes[5]} min total)",
    f"Self Feeding ({total_minutes[6]} min total)",
]

# create figure

fig, ax = plt.subplots(figsize=(12, 4), dpi=300)

# create stacked bar chart
for i in range(len(adls)):
    ax.bar(
        patients,
        values[i],
        label=labels[i],
        bottom=np.sum(values[:i], axis=0),
        width=0.8,
        alpha=0.8,
    )

# axis properties
ax.legend(loc="upper left", fancybox=True, framealpha=0.3, fontsize=14)
ax.set_ylabel("Minutes Recorded")
ax.set_xlabel("Patient ID")
ax.set_ylim(0, 250)

# make plot background transparent
fig.patch.set_alpha(0)

# show plot
plt.show()
