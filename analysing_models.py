import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        """
        # possible error on
        data_obs_4096_token_length_50_exp
        """
    )
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import re
    import dotenv

    # loads the save paths from the .env file
    save_figs = os.getenv("save_figs")
    save_appendix = os.getenv("save_appendix")

    synth_cor_stats_df = pd.read_csv("data/corruption_results.csv")
    synth_cor_stats_df["target_wer"] = (synth_cor_stats_df["target_wer"] * 100).astype(
        int
    )
    synth_cor_stats_df["target_cer"] = (synth_cor_stats_df["target_cer"] * 100).astype(
        int
    )

    synth_cor_stats_df["target_cer"] = np.where(
        synth_cor_stats_df["target_cer"] == 0, 5, synth_cor_stats_df["target_cer"]
    )
    return (
        dotenv,
        np,
        os,
        pd,
        plt,
        re,
        save_appendix,
        save_figs,
        sns,
        synth_cor_stats_df,
    )


@app.cell
def __(synth_cor_stats_df):
    synth_cor_stats_df
    return


@app.cell
def __(os, pd):
    def process_experiment_results(folder_path, min_cer=None, max_cer=None):
        _cer_vals_df = []

        for _file in os.listdir(folder_path):
            _temp = pd.read_csv(os.path.join(folder_path, _file))

            # Filter rows based on CER range if specified
            if min_cer is not None:
                _temp = _temp[_temp["cer_orig"] >= min_cer]
            if max_cer is not None:
                _temp = _temp[_temp["cer_orig"] <= max_cer]

            # Only proceed if there are rows left after filtering
            if not _temp.empty:
                counts = _temp.shape[0]
                _temp = (
                    _temp[["wer", "cer", "wer_orig", "cer_orig", "erp_cer", "erp_wer"]]
                    .median()
                    .to_frame()
                    .T
                )
                _temp["total_obs"] = counts
                _temp["model"] = _file.replace(".csv", "")
                _cer_vals_df.append(_temp)

        if not _cer_vals_df:
            return (
                pd.DataFrame()
            )  # Return empty DataFrame if no data meets the criteria

        _cer_vals_df = pd.concat(_cer_vals_df, ignore_index=True)

        _cer_vals_df[["target_cer", "target_wer"]] = _cer_vals_df["model"].str.extract(
            r"cer_(\d*)_wer_(\d*)"
        )

        # Convert the extracted values to integers
        _cer_vals_df["target_cer"] = _cer_vals_df["target_cer"].astype(int)
        _cer_vals_df["target_wer"] = _cer_vals_df["target_wer"].astype(int)

        return _cer_vals_df

    def process_single_experiment_result(file_path, min_cer=None, max_cer=None):
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the CSV file
        _temp = pd.read_csv(file_path)

        # Filter rows based on CER range if specified
        if min_cer is not None:
            _temp = _temp[_temp["cer_orig"] >= min_cer]
        if max_cer is not None:
            _temp = _temp[_temp["cer_orig"] <= max_cer]

        # If no data meets the criteria, return an empty DataFrame
        if _temp.empty:
            return pd.DataFrame()

        # Calculate median values
        result = _temp[["wer", "cer", "wer_orig", "cer_orig"]].median().to_frame().T

        # Add model name
        file_name = os.path.basename(file_path)
        result["model"] = file_name.replace(".csv", "")
        result["total_obs"] = _temp.shape[0]

        return result

    return process_experiment_results, process_single_experiment_result


@app.cell
def __(mo):
    mo.md(
        r"""
        # Analysing Model performance

        This notebook is to model the performance of the LLM's that have been trained to try and visual how well they are working and find strategies that do better
        """
    )
    return


@app.cell
def __(np):
    def wer_cer_loss(x, y, x_ref1, y_ref1, x_ref2, y_ref2):
        # Normalize the coordinates
        x_norm = (x - x_ref1) / (x_ref2 - x_ref1)
        y_norm = (y - y_ref1) / (y_ref2 - y_ref1)

        # Set negative normalized values to zero
        x_norm_star = max(0, x_norm)
        y_norm_star = max(0, y_norm)

        # Calculate the loss
        loss = np.sqrt(x_norm_star**2 + y_norm_star**2)

        return loss

    return (wer_cer_loss,)


@app.cell
def __(os, pd, wer_cer_loss):
    model_performance_df = [
        pd.DataFrame({"model": "GPT4", "wer": 0.17, "cer": 0.09}, index=[0])
    ]

    # get the original wer and cer using wer_orig and cer_orig
    temp = pd.read_csv(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv")
    )
    temp = temp[["wer_orig", "cer_orig"]].median().to_frame().T
    temp.rename(columns={"wer_orig": "wer", "cer_orig": "cer"}, inplace=True)
    temp["model"] = "original"

    model_performance_df.append(temp)

    # add base llama
    temp = pd.read_csv(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv")
    )
    temp = temp[["wer", "cer"]].median().to_frame().T
    temp["model"] = "base llama"

    model_performance_df.append(temp)

    _folder_path = "data/cer_exp/results"
    for _file in os.listdir(_folder_path):
        _temp = pd.read_csv(os.path.join(_folder_path, _file))
        _temp = _temp[["wer", "cer"]].median().to_frame().T
        _temp["model"] = _file.replace(".csv", "")

        model_performance_df.append(_temp)

    model_performance_df = pd.concat(
        model_performance_df, ignore_index=True
    ).sort_values("cer")

    model_performance_df["total_error"] = model_performance_df.apply(
        lambda row: wer_cer_loss(row["cer"], row["cer"], 0.17, 0.09, 0.41, 0.304),
        axis=1,
    )

    # synthetic_dataset_df['text'].apply(lambda x: len(tokenizer.encode(x)))
    return model_performance_df, temp


@app.cell
def __(model_performance_df):
    model_performance_df.sort_values("cer")
    return


@app.cell
def __(mo):
    mo.md(
        r"""# Creating a plot that visusalises the change in CER and WER for different models and approaches"""
    )
    return


@app.cell
def __(model_performance_df, plt, sns):
    sns.scatterplot(data=model_performance_df, x="cer", y="wer")
    plt.title("CER and WER performance relative to\nGPT4 and LLama 3.1 base instruct")

    # Get the cer and wer values for 'GPT4'
    gpt4_values = model_performance_df[model_performance_df["model"] == "GPT4"]
    gpt4_cer = gpt4_values["cer"].values[0]
    gpt4_wer = gpt4_values["wer"].values[0]

    # Get the cer and wer values for 'base'
    base_values = model_performance_df[model_performance_df["model"] == "base llama"]
    base_cer = base_values["cer"].values[0]
    base_wer = base_values["wer"].values[0]

    # Get the cer and wer values for 'base'
    original_values = model_performance_df[model_performance_df["model"] == "original"]
    original_cer = original_values["cer"].values[0]
    original_wer = original_values["wer"].values[0]

    # Add infinite vertical and horizontal lines for 'GPT4'
    plt.axvline(x=gpt4_cer, color="blue", linestyle="-", label="GPT4")
    plt.axhline(y=gpt4_wer, color="blue", linestyle="-")

    # Add infinite vertical and horizontal lines for 'base'
    plt.axvline(x=base_cer, color="red", linestyle="-", label="Base Llama")
    plt.axhline(y=base_wer, color="red", linestyle="-")

    # Create the frame relative to perfect performance
    plt.xlim([0, 0.55])
    plt.ylim([0, 0.65])
    plt.legend()
    plt.show()
    return (
        base_cer,
        base_values,
        base_wer,
        gpt4_cer,
        gpt4_values,
        gpt4_wer,
        original_cer,
        original_values,
        original_wer,
    )


@app.cell
def __(mo):
    mo.md(
        """
        cer_vals_df = []

        _folder_path = 'data/cer_exp/results'
        for _file in os.listdir(_folder_path):

            _temp = pd.read_csv(os.path.join(_folder_path, _file))
            _temp = _temp[['wer', 'cer']].median().to_frame().T
            _temp['model'] = _file.replace(".csv", "")

            cer_vals_df.append(_temp)

        cer_vals_df = pd.concat(cer_vals_df, ignore_index=True)


        cer_vals_df['target_cer'] = cer_vals_df['model'].str.replace('synth200_', "").astype(int)

        _orig_values = model_performance_df[model_performance_df['model'] == 'original']
        _orig_cer = _orig_values['cer'].values[0]
        _orig_wer = _orig_values['wer'].values[0]

        plt.axhline(y=_orig_wer, color='blue', linestyle='--')
        plt.axhline(y=_orig_cer, color='blue', linestyle='-')

        sns.lineplot(data = cer_vals_df, x = 'target_cer', y = 'wer')
        sns.lineplot(data = cer_vals_df, x = 'target_cer', y = 'cer')
        """
    )
    return


@app.cell
def __(cer_vals_df):
    cer_vals_df
    return


@app.cell
def __(model_performance_df, plt, process_experiment_results, sns):
    # Example data preparation (assuming cer_vals_df and model_performance_df are already defined)
    cer_vals_df = process_experiment_results("data/cer_exp/results", min_cer=None)
    cer_vals_df["type"] = "uniform"
    # cer_vals_df['target_cer'] = cer_vals_df['model'].str.replace('synth200_', "").astype(int)

    # cer_vals_df['type'] = 'uniform'

    _orig_values = model_performance_df[model_performance_df["model"] == "original"]
    _orig_cer = _orig_values["cer"].values[0]
    _orig_wer = _orig_values["wer"].values[0]

    # Prepare the data for plotting
    melted_df = cer_vals_df.melt(
        id_vars=["target_cer"],
        value_vars=["wer", "cer"],
        var_name="metric",
        value_name="value",
    )

    # Create the FacetGrid for separate plots
    g = sns.FacetGrid(melted_df, col="metric", sharey=False, height=4, aspect=1.5)

    # Map the lineplot onto the grid
    g.map(sns.lineplot, "target_cer", "value")

    # Add horizontal lines for each plot
    for ax, metric in zip(g.axes.flat, ["wer", "cer"]):
        if metric == "wer":
            ax.axhline(y=_orig_wer, color="blue", linestyle="--")
        elif metric == "cer":
            ax.axhline(y=_orig_cer, color="blue", linestyle="-")

    # Adjust titles and labels
    g.set_titles("{col_name}")
    g.set_axis_labels("Target", "Value")

    plt.show()
    return ax, cer_vals_df, g, melted_df, metric


@app.cell
def __(cer_vals_df):
    cer_vals_df
    return


@app.cell
def __(
    model_performance_df,
    os,
    process_experiment_results,
    process_single_experiment_result,
):
    folder_path = "data/cer_wer_exp"

    _min_cer_val = None

    cer_wer_vals_df = process_experiment_results(
        "data/cer_wer_exp", min_cer=_min_cer_val
    )
    cer_wer_vals_df["type"] = "paired"

    _orig_values = model_performance_df[model_performance_df["model"] == "original"]
    _orig_cer = _orig_values["cer"].values[0]
    _orig_wer = _orig_values["wer"].values[0]

    llama_base = process_single_experiment_result(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv"),
        min_cer=_min_cer_val,
    )

    print(
        f"Total obs {cer_wer_vals_df['total_obs'].min()}. Orginal CER: {cer_wer_vals_df['cer_orig'].min().round(2)}. Orginal WER: {cer_wer_vals_df['wer_orig'].min().round(2)}"
    )

    _type = "cer"
    _orig_llama_met = model_performance_df.loc[
        model_performance_df["model"] == "base llama", _type
    ].to_list()[0]
    _orig_base_met = model_performance_df.loc[
        model_performance_df["model"] == "original", _type
    ].to_list()[0]
    # sns.lineplot(data = cer_wer_vals_df, x ='target_wer', y = _type, hue = 'target_cer')
    # plt.axhline(y=_orig_llama_met, color='red', linestyle='--')
    # plt.axhline(y=_orig_base_met, color='blue', linestyle='--')
    # sns.heatmap(data = _cer_vals_df.pivot_table(index = 'target_wer', columns='target_cer', values = 'cer'))
    return cer_wer_vals_df, folder_path, llama_base


@app.cell
def __(cer_wer_vals_df):
    cer_wer_vals_df.sort_values("cer")  # /40.1
    return


@app.cell
def __():
    (0.172 - 0.12) / 0.172
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The distribution of the error of the NCSE dataset is binomial, with the lower distribution being right skewed and the upper distribution being more normally distributed but with the centre of mass shifted away from the lower limit.
        The mean of the cer and the cer orig are almost the same, but the medians are very different.
        """
    )
    return


@app.cell
def __(os, pd, plt, sns):
    llama_base_full = pd.read_csv(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv")
    )

    sns.histplot(data=llama_base_full, x="cer_orig")
    plt.show()

    sns.histplot(data=llama_base_full.loc[llama_base_full["cer_orig"] > 0.3], x="cer")
    plt.show()

    llama_base_full[["cer", "cer_orig"]].median()
    return (llama_base_full,)


@app.cell
def __(llama_base_full, sns):
    sns.scatterplot(data=llama_base_full, x="cer_orig", y="wer_orig")
    return


@app.cell
def __(mo):
    mo.md(
        """
        # Big picture interpretation

        low cer generally does better overall. However, what is notable is that, the observed CER is between 0.2 and 0.6 in 9 of the top 10 results.
        """
    )
    return


@app.cell
def __(
    cer_vals_df,
    cer_wer_vals_df,
    llama_base,
    os,
    pd,
    plt,
    save_figs,
    sns,
    synth_cor_stats_df,
):
    _temp = pd.concat(
        [cer_wer_vals_df, cer_vals_df.loc[cer_vals_df["target_cer"] <= 40]],
        ignore_index=True,
    )
    _temp = _temp.merge(
        synth_cor_stats_df[["target_wer", "target_cer", "observed_effective_cer"]],
        on=["target_wer", "target_cer"],
    )
    _temp.rename(columns={"observed_effective_cer": "o_cer"}, inplace="True")
    _temp["erp_cer"] = (_temp["cer_orig"] - _temp["cer"]) / _temp["cer_orig"]
    _temp["erp_wer"] = (_temp["wer_orig"] - _temp["wer"]) / _temp["wer_orig"]

    sns.scatterplot(
        data=_temp, x="cer", y="wer", hue="target_cer", palette="viridis", style="type"
    )

    _orig_cer = cer_wer_vals_df["cer_orig"].min()
    _orig_wer = cer_wer_vals_df["wer_orig"].min()

    # Add infinite vertical and horizontal lines for 'GPT4'
    plt.axvline(x=_orig_cer, color="red", linestyle="--")

    # plt.axhline(y=_orig_wer, color='blue', linestyle='-')
    # plt.axhline(y=original_wer, color='blue', linestyle='-')

    # Add infinite vertical and horizontal lines for 'base'
    plt.axvline(
        x=llama_base.loc[0, "cer"], color="red", linestyle="-", label="Base Llama"
    )
    plt.axhline(y=llama_base.loc[0, "wer"], color="red", linestyle="-")
    plt.title("Performance of models trained on CER-WER pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, "over_allperformance.pdf"), dpi=300)
    plt.show()

    print(f'oringal llama cer {llama_base.loc[0,'cer']}')
    print(f'oringal llama cer {llama_base.loc[0,'wer']}')
    # _temp.sort_values('cer')

    print(f"mean CER of top 10 models {_temp['cer'].nsmallest(10).mean()}")
    print(f"mean WER of top 10 models {_temp['wer'].nsmallest(10).mean()}")
    _temp.loc[_temp["cer"] < 0.17].sort_values("cer")
    return


@app.cell
def __():
    (0.30 - 0.135) / 0.30
    return


@app.cell
def __():
    (0.41 - 0.28) / 0.41
    return


@app.cell
def __(mo):
    mo.md(
        """
        It looks like, There are two distinct drivers for reducing the overall error. When CER is very high simply getting words right is the main driver of CER reduction. However, at very low levels of CER, the occaisional very badly corrupted word can lead to replacement by a synonym or grammatically interchange word, this doesn't effect the overal WER much as the vast majority will be corrected by replacing only a few letters, however it has a substantial impact on the CER as severl new incorrect characters are introduced.

        It also looks like the model has to contain a mixture of WER-CER ratios, as when the data is split into high and low CER, the base llama outperforms all trained llama in CER. It may be that the trained llama are looking for problems that don't exist.
        """
    )
    return


@app.cell
def __(process_experiment_results):
    test = process_experiment_results("data/blend_exp", min_cer=None, max_cer=None)

    test
    return (test,)


@app.cell
def __():
    33 / 91
    return


@app.cell
def __(
    model_performance_df,
    os,
    pd,
    plt,
    process_experiment_results,
    process_single_experiment_result,
    save_figs,
    sns,
):
    _folder_path = "data/cer_wer_exp"
    _hue = "target_cer"
    # Create a figure with two subplots side by side
    _fig, (_ax2, _ax1) = plt.subplots(1, 2, figsize=(20, 8))

    balance_value = 0.17  # 0.3 is the median point

    plt.rcParams.update({"font.size": 14})
    # Plot for min_cer = 0.1, max_cer = None
    _cer_wer_vals_df1 = process_experiment_results(
        _folder_path, min_cer=balance_value, max_cer=None
    )
    _cer_wer_vals_df1["type"] = "paired"

    _cer_vals_df1 = process_experiment_results(
        "data/cer_exp/results", min_cer=balance_value
    )
    _cer_vals_df1["type"] = "uniform"
    _cer_vals_df1 = _cer_vals_df1.loc[
        _cer_vals_df1["target_cer"] <= 40
    ]  # get rid of really high values as they are junk

    sns.scatterplot(
        data=pd.concat([_cer_wer_vals_df1, _cer_vals_df1], ignore_index=True),
        x="cer",
        y="wer",
        hue=_hue,
        ax=_ax1,
        s=100,
        palette="viridis",
        style="type",
    )

    _orig_values = model_performance_df[model_performance_df["model"] == "original"]
    _orig_cer = _orig_values["cer"].values[0]
    _orig_wer = _orig_values["wer"].values[0]

    _llama_base1 = process_single_experiment_result(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv"),
        min_cer=balance_value,
        max_cer=None,
    )

    _ax1.axvline(
        x=_llama_base1.loc[0, "cer"], color="red", linestyle="-", label="Base Llama"
    )
    _ax1.axhline(y=_llama_base1.loc[0, "wer"], color="red", linestyle="-")
    _ax1.set_title(
        f"High error text\n obs {_llama_base1.loc[0,'total_obs']}, wer orig {_llama_base1.loc[0,'wer_orig'].round(2)}, cer orig {_llama_base1.loc[0,'cer_orig'].round(2)}",
        fontsize=25,
    )
    _ax1.axvline(
        x=_llama_base1.loc[0, "cer_orig"],
        color="red",
        linestyle="--",
        label="original cer",
    )
    _ax1.tick_params(axis="both", which="major", labelsize=18)

    # Plot for min_cer = None, max_cer = 0.1
    _cer_wer_vals_df2 = process_experiment_results(
        _folder_path, min_cer=None, max_cer=balance_value
    )
    _cer_wer_vals_df2["type"] = "paired"

    _cer_vals_df2 = process_experiment_results(
        "data/cer_exp/results", max_cer=balance_value
    )
    _cer_vals_df2["type"] = "uniform"
    _cer_vals_df2 = _cer_vals_df2.loc[
        _cer_vals_df2["target_cer"] <= 40
    ]  # get rid of really high values as they are junk

    sns.scatterplot(
        data=pd.concat([_cer_wer_vals_df2, _cer_vals_df2], ignore_index=True),
        x="cer",
        y="wer",
        hue=_hue,
        ax=_ax2,
        s=100,
        palette="viridis",
        style="type",
        legend=False,
    )

    _llama_base2 = process_single_experiment_result(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv"),
        min_cer=None,
        max_cer=balance_value,
    )

    _ax2.axvline(
        x=_llama_base2.loc[0, "cer"], color="red", linestyle="-", label="Base Llama"
    )
    _ax2.axhline(y=_llama_base2.loc[0, "wer"], color="red", linestyle="-")
    _ax2.set_title(
        f"Low error text\n obs {_llama_base2.loc[0,'total_obs']}, wer orig {_llama_base2.loc[0,'wer_orig'].round(2)}, cer orig {_llama_base2.loc[0,'cer_orig'].round(2)}",
        fontsize=25,
    )
    _ax2.axvline(
        x=_llama_base2.loc[0, "cer_orig"],
        color="red",
        linestyle="--",
        label="original cer",
    )
    _ax2.tick_params(axis="both", which="major", labelsize=18)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, "high_low_corruption.pdf"), dpi=300)
    plt.show()
    return (balance_value,)


@app.cell
def __(synth_cor_stats_df):
    synth_cor_stats_df
    return


@app.cell
def __(plt, process_experiment_results, sns, synth_cor_stats_df):
    _folder_path = "data/cer_wer_exp"
    _hue = "o_cer"  # 'target_cer'

    plt.rcParams.update({"font.size": 14})
    # Plot for min_cer = 0.1, max_cer = None
    _cer_wer_vals_df1 = process_experiment_results(_folder_path, max_cer=0.17)
    _cer_wer_vals_df1["type"] = "paired"
    _cer_wer_vals_df1 = _cer_wer_vals_df1 = _cer_wer_vals_df1.loc[
        _cer_wer_vals_df1["cer"] < 0.0189
    ]

    _cer_wer_vals_df1 = _cer_wer_vals_df1.merge(
        synth_cor_stats_df[["target_wer", "target_cer", "observed_effective_cer"]],
        on=["target_wer", "target_cer"],
        how="left",
    )

    _cer_wer_vals_df1.rename(
        columns={"observed_effective_cer": "o_cer"}, inplace="True"
    )

    sns.scatterplot(
        data=_cer_wer_vals_df1, x="cer", y="wer", hue=_hue, s=100, palette="viridis"
    )
    plt.xticks(rotation=25)
    plt.show()
    _cer_wer_vals_df1.sort_values("cer")
    return


@app.cell
def __(plt, process_experiment_results, sns):
    _folder_path = "data/cer_wer_exp"
    _cer_wer_vals_df1 = process_experiment_results(_folder_path, max_cer=0.17)
    _cer_wer_vals_df1["smol"] = _cer_wer_vals_df1["target_cer"] < 30
    sns.scatterplot(data=_cer_wer_vals_df1, x="erp_cer", y="erp_wer", hue="smol")
    plt.show()
    _cer_wer_vals_df1.sort_values("cer")
    return


@app.cell
def __():
    10000 * 200
    return


@app.cell
def __(pd):
    general_corruption_stats_df = pd.read_csv("data/corruption_results.csv")
    return (general_corruption_stats_df,)


@app.cell
def __(general_corruption_stats_df):
    general_corruption_stats_df
    return


@app.cell
def __(os, pd):
    cer_wer_results_df = []
    _file_path = "./data/cer_wer_exp"
    for _file in os.listdir(_file_path):
        _temp = pd.read_csv(os.path.join(_file_path, _file))
        _temp["data_type"] = _file
        cer_wer_results_df.append(_temp)

    cer_wer_results_df = pd.concat(cer_wer_results_df, ignore_index=True)

    cer_wer_results_df[["target_cer", "target_wer"]] = cer_wer_results_df[
        "data_type"
    ].str.extract(r"cer_(\d{2})_wer_(\d{2})")

    cer_wer_results_df["target_cer"] = cer_wer_results_df["target_cer"].astype(int)
    cer_wer_results_df["target_wer"] = cer_wer_results_df["target_wer"].astype(int)
    return (cer_wer_results_df,)


@app.cell
def __(cer_wer_results_df):
    cer_wer_results_df
    return


@app.cell
def __(cer_wer_results_df, sns):
    sns.histplot(data=cer_wer_results_df, x="cer_orig")
    return


@app.cell
def __(cer_wer_results_df, plt, sns):
    _file_name = "artid_841530_periodical_ns_issue_vm2-ncseproduct475_page_number_4.txt"
    _file_name = "artid_751412_periodical_pc_issue_tec_01051889_page_number_8.txt"
    # _file_name = 'artid_802845_periodical_t_issue_ttw_16051868_page_number_5.txt'
    # _file_name = 'artid_494321_periodical_ewj_issue_ewj_01051860_page_number_49.txt'

    _plot_df = cer_wer_results_df.loc[
        cer_wer_results_df["file_name"] == _file_name
    ].reset_index()

    _variable = "cer"
    # Create pivot table
    _pivot = _plot_df.pivot_table(
        values=_variable, index="target_wer", columns="target_cer"
    )

    _sns_heatmap = sns.heatmap(_pivot, cmap="YlOrRd", cbar_kws={"label": _variable})
    plt.title(
        f"Original; CER {_plot_df.loc[0, 'cer_orig'].round(2)}, WER {_plot_df.loc[0, 'wer_orig'].round(2)}"
    )

    plt.show()
    return


@app.cell
def __(mo):
    mo.md(r"""# Data length""")
    return


@app.cell
def __(os, pd, re):
    def process_experiment_results_length(folder_path, min_cer=None, max_cer=None):
        _cer_vals_df = []

        for _file in os.listdir(folder_path):
            if _file.endswith(".csv"):
                _temp = pd.read_csv(os.path.join(folder_path, _file))

                # Filter rows based on CER range if specified
                if min_cer is not None:
                    _temp = _temp[_temp["cer_orig"] >= min_cer]
                if max_cer is not None:
                    _temp = _temp[_temp["cer_orig"] <= max_cer]

                # Only proceed if there are rows left after filtering
                if not _temp.empty:
                    counts = _temp.shape[0]
                    _temp = (
                        _temp[
                            ["wer", "cer", "wer_orig", "cer_orig", "erp_cer", "erp_wer"]
                        ]
                        .median()
                        .to_frame()
                        .T
                    )
                    _temp["total_obs"] = counts

                    # Extract relevant values from the file name, such as obs and token length
                    model_info = _file.replace(".csv", "")
                    _temp["model"] = model_info

                    # Extract 'obs', 'token length', etc., from the filename
                    # Assuming the structure 'data2-obs-128-token-length-200-exp'
                    extract_pattern = r"data2-obs-(\d+)-token-length-(\d+)-exp"
                    match = re.search(extract_pattern, model_info)

                    if match:
                        _temp["obs"] = int(match.group(1))
                        _temp["token_length"] = int(match.group(2))

                    _cer_vals_df.append(_temp)

        if not _cer_vals_df:
            return (
                pd.DataFrame()
            )  # Return empty DataFrame if no data meets the criteria

        _cer_vals_df = pd.concat(_cer_vals_df, ignore_index=True)

        return _cer_vals_df

    return (process_experiment_results_length,)


@app.cell
def __(math, process_experiment_results_length):
    _temp = process_experiment_results_length("./data/data_length_exp", max_cer=None)

    # _temp = _temp[_temp['cer']<0.1]
    # _temp = _temp[_temp['wer']<0.075]

    _temp["obs_power"] = _temp["obs"].apply(lambda x: math.log2(x))
    _temp["tokens"] = _temp["obs"] * _temp["token_length"]
    _temp["tokens_power"] = (_temp["tokens"] / 1000).apply(lambda x: math.log2(x))
    _temp["target_cer"] = 10
    _temp["target_wer"] = 20

    _temp.loc[_temp["token_length"] == 200].sort_values("cer")
    return


@app.cell
def __(
    os,
    plt,
    process_experiment_results_length,
    save_figs,
    sns,
    synth_cor_stats_df,
):
    import math

    _temp = process_experiment_results_length("./data/data_length_exp", max_cer=None)

    # _temp = _temp[_temp['cer']<0.1]
    # _temp = _temp[_temp['wer']<0.075]

    _temp["obs_power"] = _temp["obs"].apply(lambda x: math.log2(x))
    _temp["tokens"] = _temp["obs"] * _temp["token_length"]
    _temp["tokens_power"] = (_temp["tokens"] / 100).apply(lambda x: math.log2(x))
    _temp["target_cer"] = 10
    _temp["target_wer"] = 20

    _temp = _temp.merge(
        synth_cor_stats_df[["target_wer", "target_cer", "observed_effective_cer"]],
        on=["target_wer", "target_cer"],
    )

    _temp.rename(columns={"observed_effective_cer": "o_cer"}, inplace="True")

    # _temp.sort_values('cer')

    # _temp[['wer', 'cer', 'target_wer', 'target_cer', 'o_cer']].corr(method = 'spearman')

    _temp.loc[_temp["cer"] < 0.17].sort_values("cer")

    _pivot = _temp.pivot_table(
        values="cer",
        index="tokens",
        columns="token_length",
    )

    sns.heatmap(_pivot, cmap="YlOrRd")
    plt.show()

    # Create a figure and two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the line plot on the right without a legend
    sns.lineplot(
        data=_temp,
        x="tokens_power",
        y="cer",
        hue="token_length",
        ax=axes[0],
        legend=False,
        palette="viridis",
    )
    axes[0].set_xlabel("Number of tokens in training set where Tokens = $100\cdot2^x$")

    # Plot the boxplot on the left
    sns.boxplot(
        data=_temp,
        x="token_length",
        y="cer",
        hue="token_length",
        ax=axes[1],
        palette="viridis",
    )
    axes[1].set_xlabel("Number of tokens per observation")
    legend = axes[1].get_legend()
    legend.set_title("Tokens per obs")

    # Add a joint title for both plots
    plt.suptitle(
        "CER by tokens per observation and total tokens in dataset", fontsize=16
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, "compare_length_volume.pdf"), dpi=300)
    plt.show()
    return axes, fig, legend, math


@app.cell
def __(mo):
    mo.md(
        r"""
        # Compare data sets

        This section compares models trained on the SMH, CA and BLN600 datasets
        """
    )
    return


@app.cell
def __(os, pd):
    def process_experiment_compare(folder_path, min_cer=None, max_cer=None):
        _cer_vals_df = []

        for _file in os.listdir(folder_path):
            if _file.endswith(".csv"):
                _temp = pd.read_csv(os.path.join(folder_path, _file))

                # Filter rows based on CER range if specified
                if min_cer is not None:
                    _temp = _temp[_temp["cer_orig"] >= min_cer]
                if max_cer is not None:
                    _temp = _temp[_temp["cer_orig"] <= max_cer]

                # Only proceed if there are rows left after filtering
                if not _temp.empty:
                    counts = _temp.shape[0]
                    _temp = (
                        _temp[
                            ["wer", "cer", "wer_orig", "cer_orig", "erp_cer", "erp_wer"]
                        ]
                        .median()
                        .to_frame()
                        .T
                    )
                    _temp["total_obs"] = counts

                    # Extract relevant values from the file name, such as obs and token length
                    model_info = _file.replace(".csv", "")
                    _temp["model"] = model_info

                    _temp["dataset"] = _temp["model"].str.split("_", n=1, expand=True)[
                        0
                    ]

                    _cer_vals_df.append(_temp)

        if not _cer_vals_df:
            return (
                pd.DataFrame()
            )  # Return empty DataFrame if no data meets the criteria

        _cer_vals_df = pd.concat(_cer_vals_df, ignore_index=True)

        return _cer_vals_df

    return (process_experiment_compare,)


@app.cell
def __(
    cer_wer_vals_df,
    llama_base,
    os,
    pd,
    plt,
    process_experiment_compare,
    process_experiment_results,
    save_figs,
    sns,
):
    _max_cer = None
    _min_cer = None

    other_datasets = process_experiment_compare(
        "data/compare_datasets_exp/", min_cer=_min_cer, max_cer=_max_cer
    )

    _cer_wer_vals_df = process_experiment_results(
        "data/cer_wer_exp", min_cer=_min_cer, max_cer=_max_cer
    )
    # _temp[['wer', 'cer', 'target_wer', 'target_cer', 'o_cer']].corr(method = 'spearman')
    comparison_synth = _cer_wer_vals_df.loc[
        (_cer_wer_vals_df["target_cer"] == 10) & (_cer_wer_vals_df["target_wer"] == 20)
    ]
    comparison_synth["dataset"] = "synthetic"

    pd.DataFrame({"dataset": "Opus (SOTA)", "cer": 0.07, "wer": 0.15}, index=[0])
    combined_dataset = pd.concat(
        [comparison_synth, other_datasets], ignore_index=True
    ).sort_values("model")

    combined_dataset["model"] = combined_dataset["dataset"]

    sns.scatterplot(data=combined_dataset, x="cer", y="wer", hue="model", s=100)

    _orig_cer = cer_wer_vals_df["cer_orig"].min()
    _orig_wer = cer_wer_vals_df["wer_orig"].min()

    # Add infinite vertical and horizontal lines for 'GPT4'
    plt.axvline(x=_orig_cer, color="red", linestyle="--")

    # plt.axhline(y=_orig_wer, color='blue', linestyle='-')
    # plt.axhline(y=original_wer, color='blue', linestyle='-')

    # Add infinite vertical and horizontal lines for 'base'
    plt.axvline(
        x=llama_base.loc[0, "cer"], color="red", linestyle="-", label="Base Llama"
    )
    plt.axhline(y=llama_base.loc[0, "wer"], color="red", linestyle="-")
    plt.title("Comparing synthetic data with real data")
    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, "compare_models.pdf"), dpi=300)
    plt.show()
    return combined_dataset, comparison_synth, other_datasets


@app.cell
def __(
    os,
    pd,
    plt,
    process_experiment_compare,
    process_experiment_results,
    process_single_experiment_result,
    save_figs,
    sns,
):
    # Create a figure with two subplots side by side
    _fig, (_ax2, _ax1) = plt.subplots(1, 2, figsize=(20, 8))

    _balance_value = 0.17
    plt.rcParams.update({"font.size": 14})

    _other_datasets_high = process_experiment_compare(
        "data/compare_datasets_exp/", min_cer=_balance_value, max_cer=None
    )
    _cer_wer_vals_df_high = process_experiment_results(
        "data/cer_wer_exp", min_cer=_balance_value, max_cer=None
    )

    _comparison_synth_high = _cer_wer_vals_df_high.loc[
        (_cer_wer_vals_df_high["target_cer"] == 10)
        & (_cer_wer_vals_df_high["target_wer"] == 20)
    ]
    _comparison_synth_high["dataset"] = "synthetic"

    _combined_dataset_high = pd.concat(
        [_comparison_synth_high, _other_datasets_high], ignore_index=True
    ).sort_values("model")
    _combined_dataset_high["model"] = _combined_dataset_high["dataset"]

    # Process data for low corruption (min_cer = None, max_cer = 0.17)
    _other_datasets_low = process_experiment_compare(
        "data/compare_datasets_exp/", min_cer=None, max_cer=_balance_value
    )
    _cer_wer_vals_df_low = process_experiment_results(
        "data/cer_wer_exp", min_cer=None, max_cer=_balance_value
    )

    _comparison_synth_low = _cer_wer_vals_df_low.loc[
        (_cer_wer_vals_df_low["target_cer"] == 10)
        & (_cer_wer_vals_df_low["target_wer"] == 20)
    ]
    _comparison_synth_low["dataset"] = "synthetic"

    _combined_dataset_low = pd.concat(
        [_comparison_synth_low, _other_datasets_low], ignore_index=True
    ).sort_values("model")
    _combined_dataset_low["model"] = _combined_dataset_low["dataset"]

    # Process base Llama results
    _llama_base1 = process_single_experiment_result(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv"),
        min_cer=_balance_value,
        max_cer=None,
    )
    _llama_base2 = process_single_experiment_result(
        os.path.join("data/results", "ncse_test_recovered_base_llama.csv"),
        min_cer=None,
        max_cer=_balance_value,
    )

    # Plot for high corruption
    sns.scatterplot(
        data=_combined_dataset_high, x="cer", y="wer", hue="model", ax=_ax1, s=200
    )
    _ax1.legend(loc="lower right")
    _ax1.axvline(
        x=_llama_base1.loc[0, "cer"], color="red", linestyle="-", label="Base Llama"
    )
    _ax1.axhline(y=_llama_base1.loc[0, "wer"], color="red", linestyle="-")
    _ax1.axvline(
        x=_llama_base1.loc[0, "cer_orig"],
        color="red",
        linestyle="--",
        label="Original CER",
    )
    _ax1.set_title(
        f"High error text\n obs {_llama_base1.loc[0,'total_obs']}, wer orig {_llama_base1.loc[0,'wer_orig'].round(2)}, cer orig {_llama_base1.loc[0,'cer_orig'].round(2)}",
        fontsize=25,
    )
    _ax1.tick_params(axis="both", which="major", labelsize=18)

    # Plot for low corruption
    sns.scatterplot(
        data=_combined_dataset_low,
        x="cer",
        y="wer",
        hue="model",
        ax=_ax2,
        s=200,
        legend=False,
    )
    _ax2.axvline(
        x=_llama_base2.loc[0, "cer"], color="red", linestyle="-", label="Base Llama"
    )
    _ax2.axhline(y=_llama_base2.loc[0, "wer"], color="red", linestyle="-")
    _ax2.axvline(
        x=_llama_base2.loc[0, "cer_orig"],
        color="red",
        linestyle="--",
        label="Original CER",
    )
    _ax2.set_title(
        f"Low error text\n obs {_llama_base2.loc[0,'total_obs']}, wer orig {_llama_base2.loc[0,'wer_orig'].round(2)}, cer orig {_llama_base2.loc[0,'cer_orig'].round(2)}",
        fontsize=25,
    )
    _ax2.tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()

    plt.savefig(os.path.join(save_figs, "high_low_corruption_comparison.pdf"), dpi=300)

    plt.show()
    return


@app.cell
def __(combined_dataset):
    combined_dataset
    return


@app.cell
def __(cer_wer_results_df):
    _file_name = "artid_494321_periodical_ewj_issue_ewj_01051860_page_number_49.txt"
    _temp = cer_wer_results_df.loc[
        cer_wer_results_df["file_name"] == _file_name
    ].reset_index()

    _temp.loc[(_temp["target_cer"] == 40) & (_temp["target_wer"] == 40), "clocrc_text"]
    return


@app.cell
def __(cer_wer_results_df):
    _file_name = "artid_494321_periodical_ewj_issue_ewj_01051860_page_number_49.txt"
    _temp = cer_wer_results_df.loc[
        cer_wer_results_df["file_name"] == _file_name
    ].reset_index()

    _temp.loc[(_temp["target_cer"] == 5) & (_temp["target_wer"] == 40), "gt_text"]
    return


if __name__ == "__main__":
    app.run()
