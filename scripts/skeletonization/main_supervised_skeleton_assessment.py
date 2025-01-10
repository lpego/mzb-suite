# %%
import argparse
import os, sys
import yaml

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path

from mzbsuite.utils import cfg_to_arguments, regression_report

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../../"  # or "../"
    PLOTS = True

# Set global configuration options
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

# %%
def main(args, cfg):
    """
    Main function to run an assessment of the length measurements.
    Computes the absolute error between manual annotations and model predictions, and reports plots grouped by species.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments parsed from the command line. Specifically:
        
            - args.input_dir: path to the directory with the model predictions
            - args.manual_annotations: path to the manual annotations
            - args.model_annotations: path to the model predictions
            - args.output_dir: path to the directory where plots accuracy report should be saved

    cfg: argparse.Namespace
        configuration options.

    Returns
    -------
    None. Plots and metrics are saved in the results directory.
    """

    # Read manual annotations
    manual_annotations = pd.read_csv(args.manual_annotations, index_col=False)
    if "Unnamed: 0" in manual_annotations.columns:
        manual_annotations = manual_annotations.drop(columns=["Unnamed: 0"])

    manual_annotations["clip_name"] = [
        "_".join(f.split("_")[:-1]) for f in manual_annotations["clip_name"]
    ]
    manual_annotations = manual_annotations.set_index("clip_name", drop=True)

    # Read model predictions for skeletons
    auto_annotations = pd.read_csv(args.model_annotations, index_col="clip_name")
    
    # Make results directory if not already present
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # merge annotations and predictions
    merged_annotations = manual_annotations.merge(
        auto_annotations, left_index=True, right_index=True, how="inner"
    )
    # Calculate absolute errors
    merged_annotations["error_body_skel"] = np.abs(
        merged_annotations["body_length"] - merged_annotations["nn_pred_body"]
    )
    merged_annotations["error_head_skel"] = np.abs(
        merged_annotations["head_length"] - merged_annotations["nn_pred_head"]
    )

    # if PLOTS: 
    # rng = np.arange(50)
    # rnd = np.random.randint(0, 10, size=(3, rng.size))
    # yrs = 1950 + rng

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.stackplot(yrs, rng + rnd, labels=['Eastasia', 'Eurasia', 'Oceania'])
    # ax.set_title('Combined debt growth over time')
    # ax.legend(loc='upper left')
    # ax.set_ylabel('Total debt')
    # ax.set_xlim(xmin=yrs[0], xmax=yrs[-1])
    # fig.tight_layout()
    # plt.show()
   
    plt.figure()
    merged_annotations.groupby("species").mean()[
        ["body_length", "error_body_skel"]
    ].plot(kind="bar", rot=90)
    plt.title("body_length_error")
    plt.savefig(args.output_dir / f"body_length_error.{cfg.glob_local_format}")
    
    plt.figure()
    merged_annotations.groupby("species").mean()[
        ["head_length", "error_head_skel"]
    ].plot(kind="bar", rot=90)
    plt.title("head_width_error")
    plt.savefig(args.output_dir / f"head_width_error.{cfg.glob_local_format}")
    
    # Scatterplot for errors of body length and colored by species
    plt.figure()
    for sp in merged_annotations["species"].unique():
        plt.scatter(
            merged_annotations.loc[merged_annotations["species"] == sp, "body_length"],
            merged_annotations.loc[merged_annotations["species"] == sp, "nn_pred_body"],
            label=sp,
        )
    plt.plot(
        [0, merged_annotations["body_length"].max()],
        [0, merged_annotations["body_length"].max()],
        "k--",
    )
    plt.legend()
    plt.xlabel("True body length (mm)")
    plt.ylabel("Predicted body lenght (mm)")
    plt.title("body_length_error_scatter")
    plt.savefig(args.output_dir / f"body_length_error_scatter.{cfg.glob_local_format}")

    # Scatterplot for errors of head width and colored by species
    plt.figure()
    for sp in merged_annotations["species"].unique():
        plt.scatter(
            merged_annotations.loc[merged_annotations["species"] == sp, "head_length"],
            merged_annotations.loc[merged_annotations["species"] == sp, "nn_pred_head"],
            label=sp,
        )

    # 1:1 line
    plt.plot(
        [0, merged_annotations["head_length"].max()],
        [0, merged_annotations["head_length"].max()],
        "k--",
    )
    plt.legend()
    plt.xlabel("True head width (mm)")
    plt.ylabel("Predicted head width (mm)")
    plt.title("head_width_error_scatter")
    plt.savefig(args.output_dir / f"head_width_error_scatter.{cfg.glob_local_format}")

    # Relative errors in %
    merged_annotations["rel_error_body_skel"] = (
        100
        * np.abs(merged_annotations["body_length"] - merged_annotations["nn_pred_body"])
        / merged_annotations["body_length"]
    )

    merged_annotations["rel_error_head_skel"] = (
        100
        * np.abs(merged_annotations["head_length"] - merged_annotations["nn_pred_head"])
        / merged_annotations["head_length"]
    )

    # barplots relative errors by species
    plt.figure()
    merged_annotations.groupby("species").mean()[["rel_error_body_skel"]].plot(
        kind="bar", rot=90
    )
    plt.ylabel("Relative error (%)")
    plt.xlabel("Species")
    plt.title("body_length_rel_error")
    plt.savefig(args.output_dir / f"body_length_rel_error.{cfg.glob_local_format}")

    plt.figure()
    merged_annotations.groupby("species").mean()[["rel_error_head_skel"]].plot(
        kind="bar", rot=90
    )
    plt.ylabel("Relative error (%)")
    plt.xlabel("Species")
    plt.title("head_width_rel_error")
    plt.savefig(args.output_dir / f"head_width_rel_error.{cfg.glob_local_format}")
    
    if PLOTS: 
        plt.show()

    plt.close("all")

    report_body = regression_report(
        y_true=merged_annotations["body_length"],
        y_pred=merged_annotations["nn_pred_body"],
        PRINT=False,
    )

    report_head = regression_report(
        y_true=merged_annotations["head_length"],
        y_pred=merged_annotations["nn_pred_head"],
        PRINT=False,
    )

    prn_str = (
        [("REPORT BODY LENGTH ESTIMATION\n", "")]
        + report_body
        + [("\nREPORT HEAD WIDTH ESTIMATION\n", "")]
        + report_head
    )
    # Write to text file the reports
    with open(args.output_dir / "estimation_report.txt", "w") as f:
        for l1, l2 in prn_str:
            if not l2:
                f.write(f"{l1}\n")
            else:
                f.write(f"{l1:>25s}: {l2: >20.3f}\n")

    return None

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_annotations", type=str, required=True)
    parser.add_argument("--manual_annotations", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    args.manual_annotations = Path(args.manual_annotations)
    args.model_annotations = Path(args.model_annotations)
    args.input_dir = Path(args.model_annotations).parents[0]
    args.output_dir = Path(args.output_dir)

    with open(args.config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = cfg_to_arguments(cfg)

    sys.exit(main(args, cfg))
