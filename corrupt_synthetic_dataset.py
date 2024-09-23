import marimo

__generated_with = "0.8.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Corrupting the synthetic dataset

        - part 1 learns the transmission distributions for the corruption model
        - part 2 corrupt synthetic dataset
        """
    )
    return


@app.cell
def __():
    import sys
    import os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import re
    import random
    from datasets import Dataset, DatasetDict
    from synthetic_data_functions import (
        get_from_wikipedia,
        process_wiki_time_line,
        process_wiki_timeline_format2,
        generate_prompts,
    )
    from scrambledtext import (
        ProbabilityDistributions,
        CorruptionEngine,
        modify_and_renormalize_probs,
    )
    from transformers import AutoTokenizer
    import evaluate

    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    )
    import seaborn as sns

    from lm_support_functions import split_text, stitch_text, inference_prompt

    print("loading cer")
    cer = evaluate.load("cer")
    print("loading cer")
    wer = evaluate.load("wer")
    return (
        AutoTokenizer,
        CorruptionEngine,
        Dataset,
        DatasetDict,
        ProbabilityDistributions,
        cer,
        evaluate,
        generate_prompts,
        get_from_wikipedia,
        inference_prompt,
        modify_and_renormalize_probs,
        np,
        os,
        pd,
        process_wiki_time_line,
        process_wiki_timeline_format2,
        random,
        re,
        sns,
        split_text,
        stitch_text,
        sys,
        tokenizer,
        tqdm,
        wer,
    )


@app.cell
def __(mo):
    mo.md("""# Part 1""")
    return


@app.cell
def __(calculate_cer, calculate_wer, pd):
    data = pd.concat(
        [
            pd.read_csv("./data/aligned/aligned_BLN600.csv"),
            pd.read_csv("./data/aligned/aligned_CA.csv"),
            pd.read_csv("./data/aligned/aligned_SMH.csv"),
        ],
        ignore_index=True,
    )

    data["cer"] = data.apply(calculate_cer, axis=1)
    data["wer"] = data.apply(calculate_wer, axis=1)

    bins = [0, 0.1, 0.2, 0.3, 0.4]
    labels = [0, 10, 20, 30]

    # Adding a new column for the binned 'cer' values
    data["binned_cer"] = pd.cut(
        data["cer"], bins=bins, labels=labels, include_lowest=True
    )
    return bins, data, labels


@app.cell
def __(data):
    gt_aligned_list = data["gt_aligned"].to_list()

    noise_aligned_list = data["noise_aligned"].to_list()

    aligned_texts = list(zip(gt_aligned_list, noise_aligned_list))
    return aligned_texts, gt_aligned_list, noise_aligned_list


@app.cell
def __(ProbabilityDistributions, aligned_texts):
    gen_probs = ProbabilityDistributions(aligned_texts)
    # save the distributions so they can be loaded into the the training script
    gen_probs.save_to_json("data/learned_corruption_distribs.json")
    return (gen_probs,)


@app.cell
def __():
    return


@app.cell
def __(gen_probs):
    print(gen_probs.deletion_counts["a"])
    print(sum(gen_probs.substitution_counts["a"].values()))
    print(sum(gen_probs.insertion_counts["a"].values()))
    print(gen_probs.character_counts["a"])
    return


@app.cell
def __(mo):
    mo.md(
        """
        # Part 2

        Corrupt the dataset. 
        The below chunk shows how a simple sentence looks for different levels of corruption
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Create an example of a corrupted sentence

        The below two chunks corrupt and expression by Ada Lovelace, the first to see here in the notebook and the second output for the paper
        """
    )
    return


@app.cell
def __(
    CorruptionEngine,
    ProbabilityDistributions,
    aligned_texts,
    gen_probs,
):
    text = "The quick brown fox jumped over the lazy goose."
    text = "We may say most aptly that the Analytical Engine weaves algebraical patterns just as the Jacquard-loom weaves flowers and leaves. "

    print(f"Correct:1.00, Subsitute:0.00, Delete:0.00, Insert:0.00    : {text}")
    for _target_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _gen_probs = ProbabilityDistributions(aligned_texts)

        demo_scrambler = CorruptionEngine(
            _gen_probs.conditional,
            _gen_probs.substitutions,
            _gen_probs.insertions,
            target_wer=1,
            target_cer=_target_prob,
        )

        # Corrupt the text and calculate CER
        corrupted_text_vals, wer_vals, cer_vals, effective_cer = (
            demo_scrambler.corrupt_text(text)
        )

        loop_joint_probs = gen_probs.calculate_joint_probabilities()
        print(
            f"Correct:{round(loop_joint_probs['correct'], 2)}, Subsitute:{round(loop_joint_probs['substitute'], 2)}, Delete:{round(loop_joint_probs['delete'], 2)}, Insert:{round(loop_joint_probs['insert'], 2)}, CER:{round(cer_vals, 2)}    : {corrupted_text_vals}"
        )
    return (
        cer_vals,
        corrupted_text_vals,
        demo_scrambler,
        effective_cer,
        loop_joint_probs,
        text,
        wer_vals,
    )


@app.cell
def __(CorruptionEngine, ProbabilityDistributions, aligned_texts, np, pd):
    _text = "We may say most aptly that the Analytical Engine weaves algebraical patterns just as the Jacquard-loom weaves flowers and leaves. "

    _results = []

    for _target_prob in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        _gen_probs = ProbabilityDistributions(aligned_texts)

        _demo_scrambler = CorruptionEngine(
            _gen_probs.conditional,
            _gen_probs.substitutions,
            _gen_probs.insertions,
            target_wer=1,
            target_cer=_target_prob,
        )

        # Corrupt the text and calculate CER
        _corrupted_text_vals, _wer_vals, _cer_vals, _effective_cer = (
            _demo_scrambler.corrupt_text(_text)
        )

        _results.append(
            {
                "Target CER": np.round(_target_prob, 2),
                "Observed CER": np.round(_cer_vals, 2),
                "Corrupted Text": _corrupted_text_vals,
            }
        )

    # Create DataFrame
    _df = pd.DataFrame(_results)

    _latex_table = """
    \\begin{table*}
    \\centering
    \\caption{As can be seen as CER increases text becomes increasingly illegible}
    \\label{tab:corruption_results}
    \\begin{tabular}{|p{0.15\\linewidth}|p{0.15\\linewidth}|p{0.7\\linewidth}|}
    \\hline
    \\textbf{Target CER} & \\textbf{Observed CER} & \\textbf{Corrupted Text} \\\\
    \\hline
    """

    # Add rows to the table
    for _, row in _df.iterrows():
        _latex_table += f"{row['Target CER']:.2f} & {row['Observed CER']:.2f} & {row['Corrupted Text']} \\\\\n\\hline\n"

    # Close the table
    _latex_table += "\\end{tabular}\n\\end{table*}"

    # Print the LaTeX table
    print(_latex_table)
    return (row,)


@app.cell
def __(mo):
    mo.md(
        """
        # Corrupting the synthetic dataset 

        The below section corrupts the dataset. This is only for illustrative purposes as the corruption is applied just before model training in the experiments.
        """
    )
    return


@app.cell
def __(CorruptionEngine, gen_probs, pd, random, tokenizer):
    # instantiate the corruption engine
    scrambler = CorruptionEngine(
        gen_probs.conditional,
        gen_probs.substitutions,
        gen_probs.insertions,
        target_wer=0.55,
        target_cer=0.17,
    )

    random.seed(1842)
    # Load the subset dataset
    synthetic_dataset_df = pd.read_parquet("./data/subset_synth_data.parquet")
    synthetic_dataset_df.rename(columns={"token_window": "text"}, inplace=True)

    (
        synthetic_dataset_df["corrupted_text"],
        synthetic_dataset_df["wer"],
        synthetic_dataset_df["cer"],
        synthetic_dataset_df["effective_cer"],
    ) = zip(
        *synthetic_dataset_df["text"].apply(lambda text: scrambler.corrupt_text(text))
    )

    synthetic_dataset_df["corrupted_tokens"] = synthetic_dataset_df[
        "corrupted_text"
    ].apply(lambda x: len(tokenizer.encode(x)))
    synthetic_dataset_df["tokens"] = synthetic_dataset_df["text"].apply(
        lambda x: len(tokenizer.encode(x))
    )
    synthetic_dataset_df["data_type"] = (
        ["training"] * 10000 + ["validation"] * 500 + ["test"] * 500
    )
    return scrambler, synthetic_dataset_df


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df[["cer", "wer", "tokens", "corrupted_tokens"]].describe()
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df.loc[0, "corrupted_text"]
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df.loc[0, "text"]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Create the NCSE test set

        This creates the NCSE test set in hugginface format such that it can be easily loaded and used by the fine-tuned LM's as the test set
        """
    )
    return


@app.cell
def __(os, pd, re):
    def load_txt_files_to_df(directory):
        """
        Loads the content of all '.txt' files within a specified directory into a pandas DataFrame.

        Parameters:
        - directory (str): The path to the directory containing '.txt' files.

        Returns:
        - pd.DataFrame: A DataFrame with a single column "article_text", where each row contains
          the content of one '.txt' file from the directory.
        """
        # Initialize a list to store the content of each text file
        content_list = []

        # Loop through each file in the directory
        for file_name in os.listdir(directory):
            # Check if the file is a '.txt' file
            if file_name.endswith(".txt"):
                file_path = os.path.join(directory, file_name)
                # Open the file and read its contents
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    content_list.append(
                        {"article_text": content, "file_name": file_name}
                    )

        # Create a DataFrame with the contents
        df = pd.DataFrame(content_list)

        return df

    def compute_metric(row, metric, prediction_col, reference_col):
        try:
            # Preprocess the text: lowercasing and replacing line breaks with spaces
            prediction = re.sub(r"\s+", " ", row[prediction_col].lower().strip())
            reference = re.sub(r"\s+", " ", row[reference_col].lower().strip())

            # Ensure the inputs to metric.compute are lists of strings
            predictions = [prediction]
            references = [reference]
            return metric.compute(predictions=predictions, references=references)
        except KeyError as e:
            print(f"KeyError: {e} in row: {row}")
            return None
        except Exception as e:
            print(f"Error: {e} in row: {row}")
            return None
    return compute_metric, load_txt_files_to_df


@app.cell
def __(Dataset, cer, compute_metric, load_txt_files_to_df, tokenizer, wer):
    ncse_df = load_txt_files_to_df("data/ncse/transcription_files").merge(
        load_txt_files_to_df("data/ncse/transcription_raw_ocr"),
        on="file_name",
        suffixes=["_gt", "_ocr"],
    )

    ncse_df.rename(
        columns={"article_text_gt": "gt_text", "article_text_ocr": "ocr_text"},
        inplace=True,
    )

    # re_order columns
    ncse_df = ncse_df.loc[:, ["file_name", "gt_text", "ocr_text"]]

    ncse_df["gt_tokens"] = ncse_df["gt_text"].apply(
        lambda text: len(tokenizer.encode(text))
    )
    ncse_df["ocr_tokens"] = ncse_df["ocr_text"].apply(
        lambda text: len(tokenizer.encode(text))
    )
    ncse_df["cer_orig"] = ncse_df.apply(
        compute_metric,
        axis=1,
        metric=cer,
        prediction_col="ocr_text",
        reference_col="gt_text",
    )
    ncse_df["wer_orig"] = ncse_df.apply(
        compute_metric,
        axis=1,
        metric=wer,
        prediction_col="ocr_text",
        reference_col="gt_text",
    )

    ncse_hf_dataset = Dataset.from_pandas(ncse_df)

    ncse_hf_dataset.save_to_disk("./data/ncse_hf_dataset")
    return ncse_df, ncse_hf_dataset


@app.cell
def __(mo):
    mo.md(
        r"""
        # Create example dataset

        Create a tiny example dataset that can be stored in github and on the lightning studio
        """
    )
    return


@app.cell
def __(Dataset, pd, scrambler, tokenizer):
    example_data_df = pd.read_parquet('data/synth_gt/synth200.parquet')

    example_data_df = example_data_df.groupby('data_type').sample(30, random_state=1865)

    example_data_df.to_parquet('data/example_data_df.parquet')

    (
        example_data_df["ocr_text"],
        example_data_df["wer_orig"],
        example_data_df["cer_orig"],
        example_data_df["effective_cer"]
    ) = zip(*example_data_df['gt_text'].apply(lambda text: scrambler.corrupt_text(text))
           )

    example_data_df["ocr_tokens"] = example_data_df["ocr_text"].apply(
        lambda text: len(tokenizer.encode(text))
    )

    example_data_df['file_name'] = example_data_df['id']

    #test set only has 5 of each type
    example_data_df_test = example_data_df.loc[example_data_df['data_type']=='test'].sample(5, random_state=1865)

    example_hf_dataset = Dataset.from_pandas(example_data_df_test)


    example_hf_dataset.save_to_disk("./data/example_hf_dataset")

    example_data_df
    return example_data_df, example_data_df_test, example_hf_dataset


@app.cell
def __(mo):
    mo.md(
        r"""
        # BLN 600

        The BLN 600 a smaller dataset
        """
    )
    return


@app.cell
def __(cer, pd, tokenizer, wer):
    BLN600_df = pd.read_csv("data/aligned/aligned_BLN600.csv")

    # BLN600_df = data
    def calculate_cer(row):
        return cer.compute(
            predictions=[row["raw_text"].lower()],
            references=[row["article_text"].lower()],
        )

    def calculate_wer(row):
        return wer.compute(
            predictions=[row["raw_text"].lower()],
            references=[row["article_text"].lower()],
        )

    # Apply the function to each row and create a new column 'cer'
    BLN600_df["cer"] = BLN600_df.apply(calculate_cer, axis=1)
    BLN600_df["wer"] = BLN600_df.apply(calculate_wer, axis=1)
    BLN600_df["tokens"] = BLN600_df["article_text"].apply(
        lambda x: len(tokenizer.encode(x))
    )
    BLN600_df[["cer", "wer", "tokens"]].describe()
    return BLN600_df, calculate_cer, calculate_wer


@app.cell
def __():
    return


@app.cell
def __(data):
    data.groupby("binned_cer").size()
    return


@app.cell
def __(BLN600_df, sns):
    sns.histplot(data=BLN600_df.loc[BLN600_df["cer"] > 0.1], x="cer", y="wer")
    return


@app.cell
def __(data, sns):
    sns.histplot(data=data.loc[data["cer"] > 0.1], x="cer", y="wer")
    return


@app.cell
def __(data, sns):
    sns.scatterplot(
        data=data.loc[data["cer"] > 0.0], x="cer", y="wer", hue="binned_cer"
    )
    return


@app.cell
def __():
    return


@app.cell
def __(data, pd):
    # Adjusting the sample fraction to 1 to include all rows (since the dataset is small)
    corruption_samples = (
        data.groupby("binned_cer")[["cer", "wer"]]
        .apply(
            lambda x: x.sample(n=2000, random_state=42, replace=True),
            include_groups=True,
        )
        .reset_index(drop=True)
    )

    # Randomly shuffle the rows of the dataframe
    corruption_samples = corruption_samples.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    corruption_samples.to_csv("./data/corruption_samples.csv")

    additional_rows = pd.DataFrame(
        {
            "wer": [1] * 2000,  # Set 'wer' to 1 for 2000 rows
            "cer": [0] * 2000,  # Set 'cer' to 0 for 2000 rows
        }
    )

    # Step 2: Concatenate the additional rows on top of the original synth_data
    corruption_samples_zero = pd.concat(
        [additional_rows, corruption_samples], ignore_index=True
    )

    corruption_samples_zero.to_csv("./data/corruption_samples_zero.csv")
    return additional_rows, corruption_samples, corruption_samples_zero


@app.cell
def __(mo):
    mo.md(r"""# Split and Stich Text""")
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df
    return


@app.cell
def __(inference_prompt, split_text, synthetic_dataset_df):
    content_text = synthetic_dataset_df.loc[0, "text"]
    _text = synthetic_dataset_df.loc[0, "corrupted_text"]

    print(f"total number of characters in text: {len(_text)}")

    split_out = [inference_prompt(x) for x in split_text(_text, n=300, m=100)]
    return content_text, split_out


@app.cell
def __(split_out):
    len(split_out)
    return


@app.cell
def __(split_out):
    split_out[4]
    return


@app.cell
def __():
    combined_list = [
        "Group theory, real men like us are breaking our backs just to scrape enough to get bread, not to get rich. Not that they care, mind you. They'd rather puzzle over some futile equation than think for one moment about the misery around them, or rely on the Chartist resolutions whose growing fear",
        "The newspapers at home more than sufficiently expose the misery around them, and rightly so. We demand fair representation, a tree democracy, (a right, not a privilege) where the common man's voice is heard. But no, they prate about their scholarly nonsense, mere meanings and insignificant news, whereas we propose new ways to...",
        "Card of a polite gentleman, offering to remove the lot of actual unhappy beings, they are rather fond of pontificating about numbers and symbols; These are the sort of people who would laugh in our faces were we to dare mention our plight in their ivy-coated halls. They do not know the meaning of hard work and hunger. For they dwell in a separate realm",
    ]
    return (combined_list,)


@app.cell
def __(combined_list, stitch_text):
    stitch_text(combined_list)
    return


@app.cell
def __(
    cer,
    combined_list,
    content_text,
    stitch_text,
    synthetic_dataset_df,
):
    content_text

    cer.compute(
        predictions=[stitch_text(combined_list).lower()],
        references=[synthetic_dataset_df.loc[0, "text"].lower()],
    )
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df.loc[0, "text"]
    return


@app.cell
def __():
    return


@app.cell
def __(difflib):
    def stitch_text2(chunks: list):
        """
        Stitches the chunks of text back together using an optimized Longest Common Subsequence (LCS) method.

        Args:
        - chunks: A list of strings, where each string is a chunk of recovered text.

        Returns:
        - The fully stitched text.
        """
        if not chunks:
            return ""

        stitched_text = chunks[0]  # Start with the first chunk

        for i in range(1, len(chunks)):
            prev_chunk = stitched_text
            curr_chunk = chunks[i]

            # Calculate the max length we want to check for overlap
            max_overlap_len = min(len(prev_chunk), len(curr_chunk))

            # Use difflib to find the longest matching subsequence within the bounds
            s = difflib.SequenceMatcher(None, prev_chunk[-max_overlap_len:], curr_chunk)
            match = s.find_longest_match(0, max_overlap_len, 0, len(curr_chunk))

            # Check if the match is meaningful
            if match.size > 0 and match.a >= len(prev_chunk) - max_overlap_len:
                # If there's a meaningful overlap, append the non-overlapping part of the current chunk
                stitched_text += curr_chunk[match.b + match.size :]
            else:
                # If no meaningful overlap is found, concatenate the entire current chunk
                stitched_text += curr_chunk

        return stitched_text
    return (stitch_text2,)


@app.cell
def __(combined_list, stitch_text2):
    stitch_text2(combined_list)
    return


@app.cell
def __():
    from difflib import SequenceMatcher

    def find_best_overlap(str1, str2, min_overlap=10, max_overlap=100):
        # First, try exact matching
        for overlap in range(min(len(str1), max_overlap), min_overlap - 1, -1):
            if str1[-overlap:] == str2[:overlap]:
                return overlap, 1.0  # Perfect match

        # If no exact match, use fuzzy matching
        best_ratio = 0
        best_overlap = 0
        for overlap in range(min(len(str1), max_overlap), min_overlap - 1, -1):
            ratio = SequenceMatcher(None, str1[-overlap:], str2[:overlap]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_overlap = overlap

        return best_overlap, best_ratio

    def stitch_text_blocks(text_blocks, min_overlap=10, max_overlap=100, min_ratio=0.8):
        if len(text_blocks) <= 1:
            return "".join(text_blocks)

        result = text_blocks[0]
        for i in range(1, len(text_blocks)):
            prev_block = result
            curr_block = text_blocks[i]

            overlap, ratio = find_best_overlap(
                prev_block, curr_block, min_overlap, max_overlap
            )

            if ratio >= min_ratio:
                # Use fuzzy matching to find the best cut point
                matcher = SequenceMatcher(
                    None, prev_block[-overlap:], curr_block[:overlap]
                )
                match = matcher.find_longest_match(0, overlap, 0, overlap)
                cut_point = overlap - match.a
                result = prev_block[:-cut_point] + curr_block
            else:
                # If no good overlap found, just append
                result += " " + curr_block

        return result

    # Example usage
    text_blocks = [
        "This is the first block of text with some errors,",
        "block of text with some errors. This is the second block.",
        "This is the second block. And this is the third block.",
    ]

    stitched_text = stitch_text_blocks(text_blocks)
    print(stitched_text)
    return (
        SequenceMatcher,
        find_best_overlap,
        stitch_text_blocks,
        stitched_text,
        text_blocks,
    )


@app.cell
def __(combined_list, stitch_text_blocks, synthetic_dataset_df, wer):
    wer.compute(
        predictions=[stitch_text_blocks(combined_list, 100).lower()],
        references=[synthetic_dataset_df.loc[0, "text"].lower()],
    )
    return


if __name__ == "__main__":
    app.run()
