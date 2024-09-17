def time_diff_in_minutes(start: str, end: str) -> float:
    """Calculates the difference in minutes between two times.

    Args:
        start (str): A string of time in the format "HH:MM".
        end (str): A string of time in the format "HH:MM".

    Returns:
        float: The difference in minutes as a decimal value.
    """
    start_h, start_m = start.split(":")
    end_h, end_m = end.split(":")
    start_total = int(start_h) * 60 + int(start_m)
    end_total = int(end_h) * 60 + int(end_m)
    return (end_total - start_total) / 60


def sum_durations(times: list[tuple[str, str]]) -> float:
    """Sums the durations between start and end times for each tuple.

    Args:
        times (List[Tuple[str, str]]): A list of tuples, each containing start and end times in the format "HH:MM".

    Returns:
        float: The total sum of durations in decimal hours.
    """
    return sum([time_diff_in_minutes(start, end) for start, end in times])


grooming = sum_durations(
    [
        ("0:00", "11:49"),
        ("0:00", "2:53"),
        ("9:04", "12:49"),
    ]
)

home_management = sum_durations(
    [
        ("2:53", "11:49"),
        ("0:00", "3:25"),
        ("4:53", "11:49"),
        ("0:00", "7:23"),
        ("0:00", "5:17"),
        ("0:00", "9:04"),
        ("1:00", "11:49"),
    ]
)

communication_management = sum_durations(
    [("3:25", "4:53"), ("7:58", "11:49"), ("0:00", "9:51"), ("0:00", "3:19")]
)

feeding = sum_durations(
    [
        ("5:51", "11:49"),
        ("0:00", "9:25"),
    ]
)

functional_mobility = sum_durations(
    [
        ("3:20", "11:49"),
        ("0:00", "11:49"),
        ("0:00", "11:49"),
        ("0:00", "11:49"),
        ("0:00", "11:49"),
    ]
)

# print the sum of durations for each category
print(
    f"Grooming: {grooming} min\n"
    f"Home Management: {home_management} min\n"
    f"Communication Management: {communication_management} min\n"
    f"Feeding: {feeding} min\n"
    f"Functional Mobility: {functional_mobility} min"
)
