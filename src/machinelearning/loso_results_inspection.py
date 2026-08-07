import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE_NAME = "../output_ieb2/ml/loso_ieb2_personalization_subject_metrics.csv"


def plot_mse_for_each_participant(df, output_dir, method="gp_personalization"):
    # filter results for the specified method
    df = df[df["method"] == method]

    participants = df["participant_id"]
    rmse = df["rmse"]

    # plot a bar chart for each participant, ordered by rmse
    plt.figure(figsize=(10, 6))
    sns.barplot(x="participant_id", y="rmse", data=df,
                order=df.sort_values("rmse")["participant_id"])
    plt.title(f"LOSO RMSE for each participant ({method})")

    plt.xticks(rotation=45)
    plt.tight_layout()
    rmse_plot_path = os.path.join(
        output_dir, f"loso_rmse_participant_{method}.png")
    plt.savefig(rmse_plot_path)
    print(f"Saved LOSO RMSE plot for each participant to: {rmse_plot_path}")
    plt.show()
    plt.close()


def plot_mse_for_each_participant(df, output_dir, method="gp_personalization"):
    # filter results for the specified method
    newdf = df[df["method"] == method]
    participants = newdf["participant_id"]
    rmse = newdf["rmse"]

    # loop over all participants and print the difference between s2 and s1
    newdf2 = df[df["method"] == "residual_personalization"]
    differences = []
    for participant in participants:
        diffs2_s1 = newdf2[newdf2["participant_id"]
                           == participant]["s2-s1"].values[0]
        differences.append(diffs2_s1)

    print(f"Participants: {participants}")
    print(f"RMSE: {rmse}")
    print(f"Differences between s2 and s1 for each participant: {differences}")

    # plot a chart with 2 bars for each participant: rmse and differences, ordered by rmse
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(participants))
    plt.bar(index, rmse, bar_width, label="RMSE")
    plt.bar([i + bar_width for i in index],
            differences, bar_width, label="Differences")

    plt.xlabel("Participant")
    plt.ylabel("Value")
    plt.title(f"LOSO RMSE and Differences for each participant ({method})")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    rmse_plot_path = os.path.join(
        output_dir, f"loso_rmse_participant_{method}.png")
    plt.savefig(rmse_plot_path)
    print(f"Saved LOSO RMSE plot for each participant to: {rmse_plot_path}")
    plt.show()
    plt.close()


def plot_boxplot_for_each_method(per_subject, output_dir):
    """Plot LOSO results for each method."""

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="method", y="rmse", data=per_subject)
    plt.title("LOSO RMSE by Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    rmse_plot_path = os.path.join(output_dir, "loso_rmse_boxplot.png")
    plt.savefig(rmse_plot_path)
    print(f"Saved LOSO RMSE boxplot to: {rmse_plot_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE_NAME)
    output_dir = os.path.dirname(INPUT_FILE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    plot_mse_for_each_participant(df, output_dir, method="gp_personalization")
    plot_boxplot_for_each_method(df, output_dir)
