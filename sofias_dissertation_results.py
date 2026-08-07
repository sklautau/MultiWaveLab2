'''
Executes code to generate Sofia's M. Sc. Dissertation Results
It tries to be smart to configure the project root and import paths.
It also changes the working directory to the project root so that
relative paths work as expected.
'''
from pathlib import Path
import os
import sys


def configure_project_root():
    """
    Find the project root from this file location and ensure src is in sys.path.

    Returns
    -------
    Path
        Absolute path to the src directory.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    src_path = project_root / "src"

    if not src_path.exists():
        raise RuntimeError(f"Expected src directory not found at: {src_path}")

    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    return src_path


if __name__ == "__main__":
    src_path = configure_project_root()
    print(f"Configured import path: {src_path}")
    try:
        import run_scripts.execute_experiments
        import run_scripts.run_all_pipelines
    except ImportError as exc:
        print(
            "Could not import 'run_scripts.execute_experiments'.\n"
            "Did you forget to set PYTHONPATH?"
        )
        raise exc

    print("Imports are working.")

    # cd to the project root directory
    project_root = src_path
    print(f"Changing working directory to project root: {project_root}")
    os.chdir(project_root)

    # run_scripts.run_all_pipelines.run_script(
    #    r'.\run_scripts\run_all_pipelines.py', "a")

    # run all experiments for IEB-1 dataset
    run_scripts.execute_experiments.run_script(
        r'.\run_scripts\execute_experiments.py',
        "--json", "..\\config_experiments\\input_ieb1",
        "--numbers", "0-14")

    # run all experiments for IEB-2 dataset
    run_scripts.execute_experiments.run_script(
        r'.\run_scripts\execute_experiments.py',
        "--json", "..\\config_experiments\\input_ieb2",
        "--numbers", "0-14")

    # run all experiments for IEB-3 dataset
    run_scripts.execute_experiments.run_script(
        r'.\run_scripts\execute_experiments.py',
        "--json", "..\\config_experiments\\input_ieb3",
        "--numbers", "0-14")

    # run all experiments for All-PPGs dataset
    run_scripts.execute_experiments.run_script(
        r'.\run_scripts\execute_experiments.py',
        "--json", "..\\config_experiments\\input_all_ppgs",
        "--numbers", "0-14")
