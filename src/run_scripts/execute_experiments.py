'''Conveniently execute several experiments'''
import argparse
import os
import re

from run_scripts.run_all_pipelines import run_script


def parse_number_ranges(spec: str, unique: bool = False) -> list[int]:
    """
    Parse a string such as:
        "6,7-10,15"

    into:
        [6, 7, 8, 9, 10, 15]

    Raises:
        ValueError if the specification is invalid.
    """
    numbers = []

    for token in spec.split(","):
        token = token.strip()

        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if m:
            start, end = map(int, m.groups())
            if start > end:
                raise ValueError(f"Invalid range: {token}")
            numbers.extend(range(start, end + 1))
        elif token.isdigit():
            numbers.append(int(token))
        else:
            raise ValueError(f"Invalid token: {token}")

    if unique:
        numbers = list(dict.fromkeys(numbers))

    return numbers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run several experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--json",
        dest="json_folder",
        default=r"../MultiWaveLab-Inputs/input_ieb3/",
        help="Path to the folder containing the input JSON files."
    )

    parser.add_argument(
        "--numbers",
        dest="experiment_numbers",
        default="1,2-8,9,10",
        help="Range of experiment numbers to run."
    )

    # parse command line
    args = parser.parse_args()

    experiments = parse_number_ranges(args.experiment_numbers, unique=True)

    for experiment_number in experiments:
        json_file = os.path.join(
            args.json_folder, f"exp{experiment_number}.json")
        if not os.path.isfile(json_file):
            print(
                f"JSON file for experiment {experiment_number} does not exist: {json_file}")
            continue

        print(
            f"Running experiment {experiment_number} with JSON file: {json_file}")
        run_script(r'.\run_scripts\run_all_pipelines.py', json_file)
