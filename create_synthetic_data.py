import marimo

__generated_with = "0.7.19"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("""
    # Creating synthetic corruption

    There doesn't seem to be enough examples of corrupted data to train models

    - part 1 learns the transmission distributions for the corruption model
    - part 2 applies the corruption to text
    - part 3 creating a dataset
    """)
    return


@app.cell
def __():
    import sys
    import os
    import pandas as pd
    from tqdm import tqdm
    from scrambledtext import (initialize_counters, calculate_character_distribution,calculate_conditional_probs,
                               generate_substitution_insertion_tables, add_default_values, update_counts, modify_and_renormalize_probs,
                               calculate_joint_probabilities, CorruptionEngine)
    from tqdm import tqdm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    return (
        AutoTokenizer,
        CorruptionEngine,
        add_default_values,
        calculate_character_distribution,
        calculate_conditional_probs,
        calculate_joint_probabilities,
        generate_substitution_insertion_tables,
        initialize_counters,
        modify_and_renormalize_probs,
        os,
        pd,
        sys,
        tokenizer,
        tqdm,
        update_counts,
    )


@app.cell
def __(mo):
    mo.md("# Part 1")
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
    # Example aligned text pairs
    aligned_texts = [
        ("New Yo@rk is big", "Nev Yo rk@is@@@"),
        ("New Yo@rk is big", "New Yo rk@is@@@@"),
        # Add more aligned text pairs here
    ]

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
def __(
    calculate_joint_probabilities,
    character_distribution,
    conditional_probs,
    modify_and_renormalize_probs,
):
    conditional_probs2 = modify_and_renormalize_probs(conditional_probs, column = 'correct', factor = .9)

    calculate_joint_probabilities(conditional_probs2, character_distribution)
    return conditional_probs2,


@app.cell
def __(mo):
    mo.md("# Part 2")
    return


@app.cell
def __(
    CorruptionEngine,
    conditional_probs2,
    insertion_table,
    substitution_table,
):
    scrambler = CorruptionEngine(conditional_probs2, substitution_table,  insertion_table)


    text = 'The quick brown fox jumped over the lazy goose.'

    print(scrambler.corrupt_text(text))
    return scrambler, text


if __name__ == "__main__":
    app.run()
