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
        # Creating synthetic dataset

        - Part 1 create a prompt template and generate 11000 prompts which will be used to create the synthetic training/test set.
        - Part 2 Generate the training set using API calls save as a single dataframe
        - Part 3 Randomly select a 200 token portion from each response to create 10k training 1k test set.

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
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    return (
        AutoTokenizer,
        CorruptionEngine,
        add_default_values,
        calculate_character_distribution,
        calculate_conditional_probs,
        calculate_joint_probabilities,
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
        wikipediaapi,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # Part 1

        Creating a prompt format for the generation of synthetic clean text. This section downloads the Timeline of the 19th century and the Time line of British Diplomatic history pages from wikipedia and structures them into dataframes focused on the 19th century.
        """
    )
    return


@app.cell
def __(get_from_wikipedia, process_wiki_time_line):
    c19_timeline = get_from_wikipedia('scrambled_text (ucabbou@ucl.ac.uk)', "Timeline_of_the_19th_century", language = 'en')

    c19_timeline = process_wiki_time_line(c19_timeline)

    c19_timeline = c19_timeline.iloc[:c19_timeline[c19_timeline['content'].str.contains('References', case=True)].index[0],:]
    return c19_timeline,


@app.cell
def __(c19_timeline):
    c19_timeline
    return


@app.cell
def __(get_from_wikipedia, process_wiki_timeline_format2):
    brit_diplo_timeline = get_from_wikipedia('scrambled_text (ucabbou@ucl.ac.uk)', "Timeline_of_British_diplomatic_history", language = 'en')
    brit_diplo_timeline = process_wiki_timeline_format2(brit_diplo_timeline)

    # remove the bibliography and so as this is not part of the timeline
    brit_diplo_timeline = brit_diplo_timeline.iloc[:brit_diplo_timeline[brit_diplo_timeline['content'].str.contains('biblio', case=False, na=False)].index[0],:]
    # subset timeline to only the 19th century
    brit_diplo_timeline = brit_diplo_timeline.loc[(brit_diplo_timeline['year']>1799) & (brit_diplo_timeline['year']<1900)]
    return brit_diplo_timeline,


@app.cell
def __(brit_diplo_timeline):
    brit_diplo_timeline
    return


@app.cell
def __():
    text_type = [
        "newspaper article",
        "obituary",
        "editorial",
        "public speech",
        "book paragraph",
        "pamphlet",
        "news report",
        "letter to the editor",
        "personal diary entry"
    ]

    writing_style = [
        "formal",
        "informal",
        "satirical",
        "religious",
        "polemic",
        "romantic",
        "persuasive",
        "descriptive"
    ]

    audience = [
        "general public",
        "scholars",
        "women",
        "reactionaries",
        "progressives",
        "military officers",
        "political leaders",
        "industrialists and merchants",
        "literary critics",
        "clergy"
    ]

    sentiment = [
        "somber",
        "optimistic",
        "neutral",
        "pessimistic",
        "enthusiastic",
        "critical",
        "hopeful",
        "nostalgic",
        "angry",
        "reflective"
    ]

    complexity = [
        "simple",
        "moderate",
        "advanced",
        "elaborate"
    ]
    return audience, complexity, sentiment, text_type, writing_style


@app.cell
def __(
    audience,
    brit_diplo_timeline,
    c19_timeline,
    complexity,
    generate_prompts,
    pd,
    sentiment,
    text_type,
    writing_style,
):
    test = generate_prompts(pd.concat([c19_timeline,brit_diplo_timeline], ignore_index = True), text_type, writing_style, audience, sentiment, complexity, num_samples=11000, word_count=500)
    test.loc[11, 'full_prompt']
    return test,


@app.cell
def __(mo):
    mo.md(r"""# Part 2: Create generate the dataset using API calls.""")
    return


if __name__ == "__main__":
    app.run()
