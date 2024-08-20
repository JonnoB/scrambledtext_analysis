import marimo

__generated_with = "0.7.20"
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
    import random
    from datasets import Dataset, DatasetDict
    from synthetic_data_functions import (get_from_wikipedia, process_wiki_time_line, process_wiki_timeline_format2,
    generate_prompts)
    from scrambledtext import (ProbabilityDistributions, CorruptionEngine, WERBasedCorruptionEngine, modify_and_renormalize_probs)
    from tqdm import tqdm
    from transformers import AutoTokenizer
    import evaluate
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

    print('loading cer')
    cer = evaluate.load('cer')
    print('loading cer')
    wer = evaluate.load('wer')
    return (
        AutoTokenizer,
        CorruptionEngine,
        Dataset,
        DatasetDict,
        ProbabilityDistributions,
        WERBasedCorruptionEngine,
        cer,
        evaluate,
        generate_prompts,
        get_from_wikipedia,
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
        wer,
    )


@app.cell
def __(mo):
    mo.md("""# Part 1""")
    return


@app.cell
def __(pd):
    data = pd.concat([pd.read_csv('./data/aligned/aligned_BLN600.csv'),
                     pd.read_csv('./data/aligned/aligned_CA.csv'),
                     pd.read_csv('./data/aligned/aligned_SMH.csv')], 
                     ignore_index=True)
    return data,


@app.cell
def __(data):
    gt_aligned_list = data['gt_aligned'].to_list()

    noise_aligned_list = data['noise_aligned'].to_list()

    aligned_texts = list(zip(gt_aligned_list, noise_aligned_list))
    return aligned_texts, gt_aligned_list, noise_aligned_list


@app.cell
def __(ProbabilityDistributions, aligned_texts):
    gen_probs = ProbabilityDistributions(aligned_texts)
    #save the distributions so they can be loaded into the the training script
    gen_probs.save_to_json('data/learned_corruption_distribs.json')
    return gen_probs,


@app.cell
def __():
    return


@app.cell
def __(gen_probs):
    print(gen_probs.deletion_counts['a'])
    print(sum(gen_probs.substitution_counts['a'].values()))
    print(sum(gen_probs.insertion_counts['a'].values()))
    print(gen_probs.character_counts['a'])
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
def __(CorruptionEngine, ProbabilityDistributions, aligned_texts):
    text = 'The quick brown fox jumped over the lazy goose.'



    print(f"Correct:1.00, Subsitute:0.00, Delete:0.00, Insert:0.00    : {text}")
    for _target_prob in [1, 0.95, 0.9, 0.85, 0.8]:

        _gen_probs = ProbabilityDistributions(aligned_texts)

        _gen_probs.modify_and_renormalize_probs(column='correct', desired_value= _target_prob, inplace=True)

        demo_scrambler = CorruptionEngine(_gen_probs.conditional, 
                                          _gen_probs.substitutions,
                                         _gen_probs.insertions)

        # Corrupt the text and calculate CER
        corrupted_text_vals, cer_vals = demo_scrambler.corrupt_text(text)

        loop_joint_probs = _gen_probs.calculate_joint_probabilities()
        print(f"Correct:{round(loop_joint_probs['correct'], 2)}, Subsitute:{round(loop_joint_probs['substitute'], 2)}, Delete:{round(loop_joint_probs['delete'], 2)}, Insert:{round(loop_joint_probs['insert'], 2)}, CER:{round(cer_vals, 2)}    : {corrupted_text_vals}")
    return (
        cer_vals,
        corrupted_text_vals,
        demo_scrambler,
        loop_joint_probs,
        text,
    )


@app.cell
def __(WERBasedCorruptionEngine, gen_probs, pd, random, tokenizer):
    #instantiate the corruption engine
    scrambler = WERBasedCorruptionEngine(gen_probs.conditional, 
                                         gen_probs.substitutions,  
                                         gen_probs.insertions)

    random.seed(1842)
    #Load the subset dataset
    synthetic_dataset_df = pd.read_parquet('./data/subset_synth_data.parquet')
    synthetic_dataset_df.rename(columns={'token_window':'text'}, inplace=True)

    synthetic_dataset_df['corrupted_text'], synthetic_dataset_df['wer'], synthetic_dataset_df['cer'] = zip(
        *synthetic_dataset_df['text'].apply(lambda text:scrambler.corrupt_text_with_wer_cer(text, target_wer = 0.55, target_cer = 0.17))
    )

    synthetic_dataset_df['corrupted_tokens'] = synthetic_dataset_df['corrupted_text'].apply(lambda x: len(tokenizer.encode(x)))
    synthetic_dataset_df['tokens'] = synthetic_dataset_df['text'].apply(lambda x: len(tokenizer.encode(x)))
    synthetic_dataset_df['data_type'] = ['training'] * 10000 + ['validation'] * 500 + ['test'] * 500
    return scrambler, synthetic_dataset_df


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df[['cer', 'wer', 'tokens','corrupted_tokens']].describe()
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df.loc[0, 'corrupted_text']
    return


@app.cell
def __(synthetic_dataset_df):
    synthetic_dataset_df.loc[0, 'text']
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Create dataset

        Create the huggingface dataset dictionary and save
        """
    )
    return


@app.cell
def __(Dataset, DatasetDict, synthetic_dataset_df):
    hf_dataset = Dataset.from_pandas(synthetic_dataset_df)

    # Split the dataset based on the 'data_type' column into training, validation, and test sets
    dataset_dict = DatasetDict({
        'train': hf_dataset.filter(lambda example: example['data_type'] == 'training'),
        'validation': hf_dataset.filter(lambda example: example['data_type'] == 'validation'),
        'test': hf_dataset.filter(lambda example: example['data_type'] == 'test')
    })

    # Optionally, save the dataset to disk for later use
    dataset_dict.save_to_disk('./data/hf_synthetic_dataset')
    return dataset_dict, hf_dataset


@app.cell
def __(WERBasedCorruptionEngine, corruption_distribs, text):
    target_text = "This is a test sentence."
    target_wer = 0.55  # 20% word error rate
    target_cer = 0.2  # 10% character error rate


    engine = WERBasedCorruptionEngine(corruption_distribs['probabilities']['conditional'], 
                                      corruption_distribs['probabilities']['substitutions'], 
                                      corruption_distribs['probabilities']['insertions'])

    # Corrupt the text
    corrupted_text = engine.corrupt_text_with_wer_cer(text, target_wer, target_cer)
    print(corrupted_text)
    return corrupted_text, engine, target_cer, target_text, target_wer


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
def __():
    60*670000/1e6
    return


@app.cell
def __(synthetic_dataset_df):
    print(synthetic_dataset_df.loc[0, ['cer', 'wer']])
    synthetic_dataset_df.loc[0, 'corrupted_text']
    return


@app.cell
def __(cer, synthetic_dataset_df):
    gpt4_recover = """isn't to pursue knowledge for its own sake, with little regard for its practical value.
    As citizens, we must advocate for a balanced approach to intellectual endeavors. The arts and sciences are the twin pillars of civilization, but their pursuit must always be tempered with purpose and relevance. Group Theory, captivating though it may be in academic circles, might risk our society's immediate welfare for the sake of intellectual curiosity.

    It is essential to recognize and reward the application of reason and theory to real-world issues. We must urge our scholars to tread a path that not only enlightens a few but also uplifts many. And while Galois's Group Theory may eventually find a crucial application, we must, at this moment, cast a cautious eye on this abstract pursuit and question whether it justifies diverting valuable resources and attention.

    The time has come to reassess our priorities, to ask ourselves if the pursuit of such esoteric knowledge truly serves the greater good or if it inadvertently widens the schism between the erudite and the rest of society.
     """

    llama38b_recover = """It is not necessary to pursue knowledge for its own sake, with little regard for its practical value.

    A serious citizen, we must advise for a balanced approach to intellectual endeavors. The arts and sciences are the twin pillars of civilization, but their pursuit must always be, tempered with purpose and reason. Group Theory, captivating to many in academic circles, might risk our society's immediate welfare for the sake of intellectual curiosity.

    It is essential to recognize and reward the application of reason and theory to real-world issues. We must urge our scholars to tread a path that not only enlightens a few, but also uplifts many. And while Galois's Group Theory may so brightly find a crucial application, we must, at this moment, cast a cautious eye on this abstract or unnecessary question whether it justifies diverting valuable resources and attention.

    The time has come to reassess our priorities, to ask ourselves if the pursuit of such esoteric knowledge truly serves the greater good or if it merely widens the schism between"""

    llama370b_recover = """is (in) to pursue knowledge for its own sake, with little regard for its practical value,

    As citizens, we must advocate for a balanced approach to intellectual endeavors. The arts and sciences are the twin pillars of civilization, but their pursuit must always be tempered with purpose and relevance. Group Theory, captivating though it may be in academic circles, might risk our society's immediate welfare for the sake of intellectual curiosity.

    It is essential to recognize and reward the application of reason and theory to real-world issues. We must urge our scholars to tread a path that not only enlightens a few but also uplifts many. And while Galois's Group Theory may so elliptically find a crucial application, We must, at this moment, cast a cautious eye on this abstract pursuit and question whether it justifies diverting valuable resources and attention.

    The time has come to reassess our priorities, to ask ourselves if the pursuit of such esoteric knowledge truly serves the greater good or if it inadvertently widens the schism between the"""

    gemma29b_recover = """is) to pursue knowledge for its own sake, with little regard for its practical valud, As citizens, we must advocate for a balanced approach to intellectual endeavors. The arts and sciences are the twin pillars of civilization, but their pursuit must always be tempered with purpose and relevance. Group Theory, captivating though it may be in academic circles, might risk our society's immediate welfare for the sake of intellectual curiosity.

    It is essential to recognize and reward the application of reason and theory to the real world issues. We must urge our scholars to tread a path that not only enlightens a few but also uplifts many. And while Galois's Group Theory also well might find a crucial application, we must, at this time, cast a cautious eye on this abstract pursuit and question whether it justifies diverting valuable resources and attention.

    The time has come to reassess our priorities, to ask ourselves if the pursuit of such esoteric knowledge truly serves the greater good or if it inadvertently widens the chasm between the er"""

    cer.compute(predictions=[llama370b_recover.lower()], references=[synthetic_dataset_df.loc[0, 'text'].lower()])
    return (
        gemma29b_recover,
        gpt4_recover,
        llama370b_recover,
        llama38b_recover,
    )


@app.cell
def __(pd, tokenizer):
    guten_ht_df = pd.read_csv('./data/Guten_HT_highpairs.tsv')
    guten_ht_df['chars'] = guten_ht_df['gsent'].apply(len)
    guten_ht_df['tokens'] = guten_ht_df['gsent'].apply(lambda x: len(tokenizer.encode(x)))
    guten_ht_df[['cer', 'wer', 'chars', 'tokens']].describe()
    return guten_ht_df,


@app.cell
def __(mo):
    mo.md(
        r"""
        # The Gutenberg HT dataset 

        This is a large collection of paired sentences. Although the number of errors they have are quite low.

        Data comes from
        The Gutenberg-HathiTrust Parallel Corpus: A Real-World Dataset for Noise Investigation in Uncorrected OCR Texts
        """
    )
    return


@app.cell
def __(guten_ht_df):
    guten_ht_df.loc[(guten_ht_df['cer']>0.2) & (guten_ht_df['cer']<0.3)]
    return


@app.cell
def __(guten_ht_df):
    guten_ht_df.loc[(guten_ht_df['cer']>0.3) & (guten_ht_df['cer']<0.4)]
    return


@app.cell
def __(guten_ht_df):
    guten_ht_df[['wer', 'cer']].describe()
    return


@app.cell
def __(guten_ht_df):
    guten_ht_df['tokens'].sum()
    return


@app.cell
def __():
    (160*670000/1e6)*0.13
    return


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
def __(BLN600_df):
    BLN600_df
    return


@app.cell
def __(cer, pd, tokenizer, wer):
    BLN600_df = pd.read_csv('data/aligned/aligned_BLN600.csv')
    def calculate_cer(row):
        return cer.compute(predictions=[row['raw_text'].lower()], references=[row['article_text'].lower()])
    def calculate_wer(row):
        return wer.compute(predictions=[row['raw_text'].lower()], references=[row['article_text'].lower()])    

    # Apply the function to each row and create a new column 'cer'
    BLN600_df['cer'] = BLN600_df.apply(calculate_cer, axis=1)
    BLN600_df['wer'] = BLN600_df.apply(calculate_wer, axis=1)
    BLN600_df['tokens'] = BLN600_df['article_text'].apply(lambda x: len(tokenizer.encode(x)))
    BLN600_df[['cer', 'wer',  'tokens']].describe()
    return BLN600_df, calculate_cer, calculate_wer


@app.cell
def __(BLN600_df):
    BLN600_df['tokens'].sum()
    return


@app.cell
def __(BLN600_df, synthetic_dataset_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    #sns.scatterplot(data = guten_ht_df, x = 'cer', y = 'wer')
    sns.scatterplot(data= BLN600_df, x = 'cer', y = 'wer')
    sns.scatterplot(data= synthetic_dataset_df, x = 'cer', y = 'wer')
    plt.title('Comparing real wer cer ratios with randomly sampled ratios')
    return plt, sns


@app.cell
def __(sns, synthetic_dataset_df):
    #The error is normally distributed
    sns.kdeplot(data= synthetic_dataset_df, x = 'cer')
    return


@app.cell
def __(guten_ht_df, sns):
    sns.histplot(data= guten_ht_df.loc[guten_ht_df['cer']>0.1], x = 'cer', y = 'wer')
    return


@app.cell
def __(guten_ht_df):
    guten_ht_df.describe()#.loc[(guten_ht_df['cer']>0.18) & (guten_ht_df['cer']<.22), 'cer'].describe()
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # BLN600 Sequence

        These are the small sequences
        """
    )
    return


@app.cell
def __(pd, tokenizer):
    sequences_df = pd.read_csv('./data/BLN600/ocr_paper_data/train.csv')
    sequences_df['ocr_chars'] = sequences_df['OCR Text'].apply(len)
    sequences_df['ocr_tokens'] = sequences_df['OCR Text'].apply(lambda x: len(tokenizer.encode(x)))
    sequences_df['ocr_tokens_char'] = sequences_df['ocr_tokens'] /sequences_df['ocr_chars']

    sequences_df['cleaned_chars'] = sequences_df['Ground Truth'].apply(len)
    sequences_df['cleaned_tokens'] = sequences_df['Ground Truth'].apply(lambda x: len(tokenizer.encode(x)))
    sequences_df['cleaned_tokens_char'] = sequences_df['cleaned_tokens'] /sequences_df['cleaned_chars']
    sequences_df.loc[(sequences_df['CER']>0.3) & (sequences_df['CER']<0.4)]
    return sequences_df,


@app.cell
def __(sequences_df):
    # Function to convert two columns into a dictionary for each row
    pairwise_dict = sequences_df.loc[(sequences_df['CER']>0.3) & (sequences_df['CER']<0.4)].sample(3).apply(lambda row: {row['OCR Text']: row['Ground Truth']}, axis=1)

    # Convert the Series of dictionaries to a list if you prefer
    pairwise_dict.tolist()
    return pairwise_dict,


@app.cell
def __(sequences_df):
    sequences_df[['cleaned_tokens_char', 'ocr_tokens_char']].describe()
    return


@app.cell
def __():
    10/12
    return


if __name__ == "__main__":
    app.run()
