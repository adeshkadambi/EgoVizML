import pandas as pd
import numpy as np

from egoviz import mod1
from egoviz.models.processing import row_wise_min_max_scaling


def test_egoviz_package():
    assert mod1.func1() == "func1() in mod1.py"


def test_row_wise_min_max_scaling():
    # Create a sample dataframe
    df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 8, 9], "C": [3, 12, 6]})

    # Call the function
    df_scaled = row_wise_min_max_scaling(df)

    # Check if the scaling is correct
    expected_result = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0], "B": [0.5, 0.6, 1.0], "C": [1.0, 1.0, 0.5]}
    )

    assert pd.DataFrame.equals(df_scaled, expected_result)
