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
    Find the project root from this file location and ensure src is in sys.path
    and the PYTHONPATH environment variable.

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

    # 1. Update Python's internal path for the current script
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    # 2. Update the OS environment so subprocesses inherit the path
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if src_path_str not in current_pythonpath:
        os.environ["PYTHONPATH"] = f"{src_path_str}{os.pathsep}{current_pythonpath}"

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

    # 3. Use forward slashes (/) which work flawlessly on BOTH Windows and Linux

    ################################################################
    # Run all 15 experiments for each dataset (IEB-1, IEB-2, IEB-3, All-PPGs)
    ################################################################

    # run all experiments for IEB-1 dataset
    run_scripts.execute_experiments.run_script(
            './run_scripts/execute_experiments.py',
            "--json", "../config_experiments/input_ieb1",
            "--numbers", "0-14")

    # run all experiments for IEB-2 dataset
    run_scripts.execute_experiments.run_script(
            './run_scripts/execute_experiments.py',
            "--json", "../config_experiments/input_ieb2",
            "--numbers", "0-14")

    # run all experiments for IEB-3 dataset
    run_scripts.execute_experiments.run_script(
            './run_scripts/execute_experiments.py',
            "--json", "../config_experiments/input_ieb3",
            "--numbers", "0-14")

    # run all experiments for All-PPGs dataset
    run_scripts.execute_experiments.run_script(
            './run_scripts/execute_experiments.py',
            "--json", "../config_experiments/input_all_ppgs",
            "--numbers", "0-14")

    ################################################################
    # Extra plots. Some scripts require setting SHOULD_PLOT = True in the code to generate plots.
    ################################################################

    # To generate plots regarding the ECG and PPG wavefoms, including
    # PSD, histogram, edit ppg.py or ecg.py to enable plotting using
    # SHOULD_PLOT = True (this is in the code, in the beginning)
    # and run the following commands to generate the plots.
    # and then run the PPG processing script
    run_scripts.run_all_pipelines.run_script(
            r'./signal_processing/ppg.py',
            "../config_experiments/input_ieb1/exp0.json"
        )

    # another example, now with ECG and IEB-3 dataset
    # remember to first set SHOULD_PLOT = True in ecg.py to generate plots
    run_scripts.run_all_pipelines.run_script(
            r'./signal_processing/ecg.py',
            "../config_experiments/input_ieb3/exp0.json"
        )

    # to obtain statistics about SQI
    # One can choose: "--pipelines=all" but in this case all
    # pipelines were executed previously and results are available in the output folder.
    run_scripts.run_all_pipelines.run_script(
        r'./run_scripts/sqi_statistics.py',
        "--modality=ecg",
        "--pipelines=ecg_neurokit",
        "--show-plots",
        "../config_experiments/input_ieb3/exp0.json"
    )

    # to obtain statistics from segments file, including the
    # contribution of each participant.
    run_scripts.run_all_pipelines.run_script(
        r'./run_scripts/segments_statistics.py',
        "../config_experiments/input_ieb3/exp0.json",
        "--show-plots",
        "--sort-by-duration"
    )

    # outdated code, but maybe useful to obtain extra
    # statistics from segments file, including the
    # contribution of each participant. It also outputs a Latex
    # table with the results.
    run_scripts.run_all_pipelines.run_script(
        r'./signal_processing/sqi_dataframe_creation.py',
        "../config_experiments/input_ieb3/exp0.json"
    )

    # The number of features can be obtained with script
    run_scripts.run_all_pipelines.run_script(
        r'./run_scripts/features_csv_statistics.py',
        "../../multiwavelab_outputs/output_ieb1/features/features14_train_selected_features.csv"
    )

    # Run the modality importance study. This takes longer than previous scripts, as it runs a regression model for each modality and each feature.
    # This command will generate a text file with the results of the modality importance study.
    # The results will be saved in the specified output directory.
    # The input data is a CSV file with the selected features.
    # The output will be a text file with the modality importance scores.
    # This code writes to stdout, so we redirect the output to a text file using the ">" operator:
    # command line... > ../multiwavelab_outputs/output_ieb3/features_importance/ieb3_features14_modality_study.txt
    run_scripts.run_all_pipelines.run_script(
        r'./machinelearning/modality_importance_study.py',
        "ieb3",
        "../../multiwavelab_outputs/output_ieb3/features/features14_train_selected_features.csv"
    )

    # the evolution of the features over time can be obtained with the script below. It generates a plot for each feature,
    # showing the evolution of the feature over time for each participant.
    # This generates a lot of plots! You may want to limit the number of features to plot by editing the script.
    run_scripts.run_all_pipelines.run_script(
        r'./run_scripts/plot_feature_waveforms_by_participant.py',
        "--show-plots",
        "--standardize-features",
        "../../multiwavelab_outputs/output_ieb1/features/features14_train_selected_features.csv"
    )
