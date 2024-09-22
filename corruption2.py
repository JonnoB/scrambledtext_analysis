import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    # python training_script.py cer '{"cer":0.2}' synth_gt/synth200.parquet cer_exp

    from scrambledtext import (
        ProbabilityDistributions,
        CorruptionEngine,
        modify_and_renormalize_probs,
    )

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import time
    import evaluate
    from lm_support_functions import (
        training_prompt,
        compute_metric,
        infer_on_test_set,
        cleaning_prompt_formatter,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    import dotenv

    # loads the save paths from the .env file
    save_figs = os.getenv("save_figs")
    save_appendix = os.getenv("save_appendix")
    return (
        AutoTokenizer,
        CorruptionEngine,
        ProbabilityDistributions,
        cleaning_prompt_formatter,
        compute_metric,
        dotenv,
        evaluate,
        infer_on_test_set,
        modify_and_renormalize_probs,
        np,
        os,
        pd,
        plt,
        save_appendix,
        save_figs,
        sns,
        time,
        tokenizer,
        training_prompt,
    )


@app.cell
def __(pd):
    synth_data2 = pd.read_parquet("data/synth_gt/synth200.parquet")  # .sample(3)

    synth_data2
    return (synth_data2,)


@app.cell
def __(
    CorruptionEngine,
    ProbabilityDistributions,
    np,
    os,
    pd,
    tokenizer,
    training_prompt,
):
    ##
    ## Testing only three being used
    ##

    _file_path = "data/corruption_results.csv"
    if not os.path.exists(_file_path):
        synth_data = pd.read_parquet("data/synth_gt/synth200.parquet")

        corruption_probs = ProbabilityDistributions().load_from_json(
            "data/learned_corruption_distribs.json"
        )

        # Initialize an empty list to store the results
        results = []

        # Loop over WER and CER values in 0.1 increments
        for wer in np.arange(0.1, 1.1, 0.1):
            for cer in [0.05]:  # np.arange(0.1, 1, 0.1)
                print(f"Processing WER={wer}, CER={cer}")

                # Initialize the corruption engine for the current wer and cer
                corruption_function = CorruptionEngine(
                    corruption_probs.conditional,
                    corruption_probs.substitutions,
                    corruption_probs.insertions,
                    target_cer=cer,
                    target_wer=wer,
                )

                # Apply corruption to the text and compute the necessary statistics
                (
                    synth_data["ocr_text"],
                    synth_data["observed_wer"],
                    synth_data["observed_cer"],
                    synth_data["observed_effective_cer"],
                ) = zip(
                    *synth_data["gt_text"].apply(
                        lambda text: corruption_function.corrupt_text(text)
                    )
                )

                # Generate the full prompt and calculate total tokens
                synth_data["full_prompt"] = synth_data.apply(
                    lambda row: training_prompt(row, "ocr_text", "gt_text", tokenizer)[
                        "full_prompt"
                    ],
                    axis=1,
                )
                synth_data["total_tokens"] = synth_data["full_prompt"].apply(
                    lambda text: len(tokenizer.encode(text))
                )

                # Calculate mean, median, and max of total tokens
                mean_tokens = synth_data["total_tokens"].mean()
                median_tokens = synth_data["total_tokens"].median()
                max_tokens = synth_data["total_tokens"].max()

                # Store the results for this combination of wer and cer
                results.append(
                    {
                        "target_wer": wer,
                        "target_cer": cer,
                        "mean_tokens": mean_tokens,
                        "median_tokens": median_tokens,
                        "max_tokens": max_tokens,
                        "observed_wer": synth_data["observed_wer"].mean(),
                        "observed_cer": synth_data["observed_cer"].mean(),
                        "observed_effective_cer": synth_data[
                            "observed_effective_cer"
                        ].mean(),
                    }
                )

        # Convert results to a DataFrame for easy analysis
        results_df = pd.DataFrame(results)

        # Display the results
        print(results_df)

        results_df.to_csv(_file_path, index=False)

    else:
        results_df = pd.read_csv(_file_path)
        results_df["target_wer"] = results_df["target_wer"].round(1)
        results_df["target_cer"] = results_df["target_cer"].round(1)
    return (
        cer,
        corruption_function,
        corruption_probs,
        max_tokens,
        mean_tokens,
        median_tokens,
        results,
        results_df,
        synth_data,
        wer,
    )


@app.cell
def __(results_df):
    results_df
    return


@app.cell
def __(os, plt, results_df, save_appendix, sns):
    # Pivot the DataFrame to create a grid for the heatmap
    _heatmap_data = results_df.pivot(
        index="target_wer", columns="target_cer", values="max_tokens"
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(_heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Heatmap of Max Tokens for Each WER and CER Pair")

    plt.savefig(os.path.join(save_appendix, "cer_wer_max_tokens_grid.pdf"), dpi=300)
    plt.show()
    return


@app.cell
def __(os, save_appendix):
    os.path.join(save_appendix, "cer_effect_cer_grid.pdf")
    return


@app.cell
def __(os, plt, results_df, save_appendix, sns):
    # Pivot the DataFrame to create a grid for the heatmap
    _heatmap_data = results_df.pivot(
        index="target_wer", columns="target_cer", values="observed_effective_cer"
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(_heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Heatmap of effective CER for Each WER and CER Pair")

    plt.savefig(os.path.join(save_appendix, "cer_effect_cer_grid.pdf"), dpi=300)
    plt.show()
    return


@app.cell
def __():
    return


@app.cell
def __():
    2**8
    return


if __name__ == "__main__":
    app.run()
