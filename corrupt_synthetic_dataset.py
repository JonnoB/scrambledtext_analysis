import marimo

__generated_with = "0.7.19"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        """
        # Corrupting the synthetic dataset

        - part 1 learns the transmission distributions for the corruption model
        - part 2 corrupt synthetic dataset
        - part 3 Measure KL divergence for different levels of corruption
        - part 4 Word level corruption
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
    import wikipediaapi
    import random
    from synthetic_data_functions import (get_from_wikipedia, process_wiki_time_line, process_wiki_timeline_format2,
    generate_prompts)
    from scrambledtext import (initialize_counters, calculate_character_distribution,calculate_conditional_probs,
                               generate_substitution_insertion_tables, add_default_values, update_counts, modify_and_renormalize_probs,
                               calculate_joint_probabilities, CorruptionEngine)
    from tqdm import tqdm
    from transformers import AutoTokenizer
    import evaluate
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

    cer = evaluate.load('cer')
    wer = evaluate.load('wer')
    return (
        AutoTokenizer,
        CorruptionEngine,
        add_default_values,
        calculate_character_distribution,
        calculate_conditional_probs,
        calculate_joint_probabilities,
        cer,
        evaluate,
        generate_prompts,
        generate_substitution_insertion_tables,
        get_from_wikipedia,
        initialize_counters,
        modify_and_renormalize_probs,
        np,
        os,
        pd,
        process_wiki_time_line,
        process_wiki_timeline_format2,
        random,
        re,
        sys,
        tokenizer,
        tqdm,
        update_counts,
        wer,
        wikipediaapi,
    )


@app.cell
def __(mo):
    mo.md("""# Part 1""")
    return


@app.cell
def __(pd):
    data = pd.read_csv('./data/aligned/aligned_BLN600.csv')
    return data,


@app.cell
def __(data):
    gt_aligned_list = data['gt_aligned'].to_list()

    noise_aligned_list = data['noise_aligned'].to_list()
    return gt_aligned_list, noise_aligned_list


@app.cell
def __(
    add_default_values,
    calculate_character_distribution,
    calculate_conditional_probs,
    generate_substitution_insertion_tables,
    gt_aligned_list,
    initialize_counters,
    noise_aligned_list,
    update_counts,
):
    aligned_texts = list(zip(gt_aligned_list, noise_aligned_list))

    # Initialize counters
    deletion_counts, insertion_counts, substitution_counts, character_counts = initialize_counters()

    # Update counts for all aligned text pairs
    for gt, noise in aligned_texts:
        update_counts(gt, noise, deletion_counts, insertion_counts, substitution_counts, character_counts)

    # Calculate character distribution
    character_distribution = calculate_character_distribution(character_counts)

    # Calculate conditional probabilities
    conditional_probs = calculate_conditional_probs(deletion_counts, insertion_counts, substitution_counts, character_counts)

    # Generate substitution and insertion tables
    substitution_table, insertion_table = generate_substitution_insertion_tables(substitution_counts, insertion_counts, character_counts)

    # Add default values to tables
    conditional_probs, substitution_table, insertion_table = add_default_values(conditional_probs, substitution_table, insertion_table, character_distribution)
    return (
        aligned_texts,
        character_counts,
        character_distribution,
        conditional_probs,
        deletion_counts,
        gt,
        insertion_counts,
        insertion_table,
        noise,
        substitution_counts,
        substitution_table,
    )


@app.cell
def __(
    character_counts,
    deletion_counts,
    insertion_counts,
    substitution_counts,
):
    print(deletion_counts['a'])
    print(sum(substitution_counts['a'].values()))
    print(sum(insertion_counts['a'].values()))
    print(character_counts['a'])
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
def __(
    CorruptionEngine,
    calculate_joint_probabilities,
    character_distribution,
    conditional_probs,
    insertion_table,
    modify_and_renormalize_probs,
    substitution_table,
):
    text = 'The quick brown fox jumped over the lazy goose.'


    print(f"Correct:1.00, Subsitute:0.00, Delete:0.00, Insert:0.00    : {text}")
    for factor in [1, 0.95, 0.9, 0.85, 0.8]:

        demo_conditional_probs = modify_and_renormalize_probs(conditional_probs, column = 'correct', factor = factor)

        loop_joint_probs = calculate_joint_probabilities(demo_conditional_probs, character_distribution)

        demo_scrambler = CorruptionEngine(demo_conditional_probs, substitution_table,  insertion_table)

        print(f"Correct:{round(loop_joint_probs['correct'], 2)}, Subsitute:{round(loop_joint_probs['substitute'], 2)}, Delete:{round(loop_joint_probs['delete'], 2)}, Insert:{round(loop_joint_probs['insert'], 2)}    : {demo_scrambler.corrupt_text(text)}")
    return (
        demo_conditional_probs,
        demo_scrambler,
        factor,
        loop_joint_probs,
        text,
    )


@app.cell
def __(
    CorruptionEngine,
    conditional_probs,
    insertion_table,
    pd,
    random,
    substitution_table,
):
    #instantiate the corruption engine
    scrambler = CorruptionEngine(conditional_probs, substitution_table,  insertion_table)

    random.seed(1842)
    #Load the subset dataset
    synthetic_dataset_df = pd.read_parquet('./data/subset_synth_data.parquet')
    synthetic_dataset_df.rename(columns={'token_window':'text'}, inplace=True)
    synthetic_dataset_df['corrupted_text'] = synthetic_dataset_df['text'].apply(scrambler.corrupt_text)
    #synthetic_dataset_df['cer'] = synthetic_dataset_df.apply(lambda row: cer(predictions))
    return scrambler, synthetic_dataset_df


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Part 3 Measure the KL divergence

        When training Llama 3 Meta used the KL-divergence of the tokens in a document from the overall token distribution to identify "low quality text". It is worth looking at how the KL divergence between gt text and corrupted text changes to see if this can be used as a sort of proxy for quality of OCR scan.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Part 4 Word level corruption

        currently corruption is applied uniformly to the text, however it is unlikely that this is what happens on a document, this section looks at how corruption is distributed within a document. This section explores the probability that a word is corrupted for a given level of corruption, and more generally throughout the document. Is corruption concentrated is smaller areas of the document?

        ## Things to look for

        - Can the corruption be represented as a vector? Do these vectors cluster?
        - Can the corruption be represented as some sort of entropy?
        - Does this entropy affect how well recovery works?
        """
    )
    return


@app.cell
def __(cer, synthetic_dataset_df, wer):
    def calculate_cer(row):
        return cer.compute(predictions=[row['corrupted_text'].lower()], references=[row['text'].lower()])
    def calculate_wer(row):
        return wer.compute(predictions=[row['corrupted_text'].lower()], references=[row['text'].lower()])    
    df = synthetic_dataset_df.copy()
    # Apply the function to each row and create a new column 'cer'
    df['cer'] = df.apply(calculate_cer, axis=1)
    df['wer'] = df.apply(calculate_wer, axis=1)
    return calculate_cer, calculate_wer, df


@app.cell
def __(df):
    df[['cer', 'wer']]
    return


@app.cell
def __(df):
    df[['cer', 'wer']].describe()
    return


if __name__ == "__main__":
    app.run()
