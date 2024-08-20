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
    import json
    import tiktoken
    from dotenv import load_dotenv
    import openai
    from synthetic_data_functions import (get_from_wikipedia, process_wiki_time_line, process_wiki_timeline_format2,
    generate_prompts)
    from scrambledtext import (initialize_counters, calculate_character_distribution,calculate_conditional_probs,
                               generate_substitution_insertion_tables, add_default_values, update_counts, modify_and_renormalize_probs,
                               calculate_joint_probabilities, CorruptionEngine)
    from tqdm import tqdm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

    from openai import OpenAI

    client = OpenAI()
    load_dotenv()
    wiki_user_agent = os.getenv('wiki_user_agent')
    return (
        AutoTokenizer,
        CorruptionEngine,
        OpenAI,
        add_default_values,
        calculate_character_distribution,
        calculate_conditional_probs,
        calculate_joint_probabilities,
        client,
        generate_prompts,
        generate_substitution_insertion_tables,
        get_from_wikipedia,
        initialize_counters,
        json,
        load_dotenv,
        modify_and_renormalize_probs,
        np,
        openai,
        os,
        pd,
        process_wiki_time_line,
        process_wiki_timeline_format2,
        random,
        re,
        sys,
        tiktoken,
        tokenizer,
        tqdm,
        update_counts,
        wiki_user_agent,
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
def __(get_from_wikipedia, process_wiki_time_line, wiki_user_agent):
    c19_timeline = get_from_wikipedia(wiki_user_agent, "Timeline_of_the_19th_century", language = 'en')

    c19_timeline = process_wiki_time_line(c19_timeline)

    c19_timeline = c19_timeline.iloc[:c19_timeline[c19_timeline['content'].str.contains('References', case=True)].index[0],:]
    return c19_timeline,


@app.cell
def __(c19_timeline):
    c19_timeline
    return


@app.cell
def __(get_from_wikipedia, process_wiki_timeline_format2, wiki_user_agent):
    brit_diplo_timeline = get_from_wikipedia(wiki_user_agent, "Timeline_of_British_diplomatic_history", language = 'en')
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
        "obituary of a named person",
        "editorial",
        "book excerpt",
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

    #'radical' can't be used instead of chartist as it doesn't seem to generate named people as frequently
    persona = ['general public', "women's rights", 'politics', 'economics and trade', 'military', 'reactionary', 'chartist', 'clergy', 'arts and culture' ]

    sentiment = [ 'positive', 'neutral','negative']

    complexity = [
        "simple",
        "moderate",
        "advanced",
    ]
    writing_style_combinations = len(text_type)*len(writing_style)*len(persona) *len(sentiment) * len(complexity)
    print(f'Total number of writing style combinations {writing_style_combinations}')
    return (
        complexity,
        persona,
        sentiment,
        text_type,
        writing_style,
        writing_style_combinations,
    )


@app.cell
def __(
    brit_diplo_timeline,
    c19_timeline,
    complexity,
    generate_prompts,
    pd,
    persona,
    sentiment,
    text_type,
    writing_style,
):
    all_prompts_df = generate_prompts(pd.concat([c19_timeline,brit_diplo_timeline], ignore_index = True), text_type, writing_style, persona, sentiment, complexity, num_samples=11000, word_count=300, seed = 1865)

    all_prompts_df['file_name'] = "index_" + all_prompts_df.index.astype(str)

    all_prompts_df.loc[11, 'full_prompt']
    return all_prompts_df,


@app.cell
def __(all_prompts_df):
    all_prompts_df
    return


@app.cell
def __(mo):
    mo.md(r"""# Part 2: Create generate the dataset using API calls.""")
    return


@app.cell
def __(json, pd):
    def create_jsonl_file(df, model, system_content, max_tokens, output_file):
        """
        Create a JSONL file for batch jobs in OpenAI format.

        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        model (str): The model to be used in the 'body' of the JSON.
        system_content (str): The content for the system message.
        max_tokens (int): The max tokens value.
        output_file (str): The path to the output JSONL file.
        include_response (bool): Whether to include the model response in the messages list.
        """
        with open(output_file, 'w') as file:
            for _, row in df.iterrows():
                messages = [
                    {"role": "system", "content": system_content}
                ]

                messages.append({"role": "user", "content": row['full_prompt']})

                entry = {
                    "custom_id": f"{row['file_name']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens
                    }
                }
                # Write each JSON object as a separate line in the JSONL file
                file.write(json.dumps(entry) + '\n')


    def convert_batch_to_dataframe(jsonl_path):
        """
        Extract 'custom_id', 'assistant' 'content', and 'usage' from a JSON string and create a DataFrame.

        Parameters:
        dict_list (str): The input a list of dictionaries.

        Returns:
        pd.DataFrame: DataFrame containing the extracted data.
        """
        with open(jsonl_path, 'r') as file:
            dict_list = [json.loads(line) for line in file]

        extracted_data = []

        for json_object in dict_list:
            data = json_object
            custom_id = data['custom_id']
            assistant_content = data['response']['body']['choices'][0]['message']['content']
            finish_reason = data['response']['body']['choices'][0]['finish_reason']
            usage = data['response']['body']['usage']

            row_data = {
                'id': custom_id,
                'generated_content': assistant_content,
                'finish_reason': finish_reason
            }

            # Add the usage dictionary to the row data
            row_data.update(usage)

            extracted_data.append(row_data)

        return pd.DataFrame(extracted_data)
    return convert_batch_to_dataframe, create_jsonl_file


@app.cell
def __(all_prompts_df, client, create_jsonl_file, os):
    # Define the path to the JSONL file
    jsonl_file_path = './data/for_gpt.jsonl'

    # Check if the JSONL file exists
    if not os.path.exists(jsonl_file_path):
        # If the file does not exist, create it and proceed with the batch request
        create_jsonl_file(all_prompts_df, 
                          model='gpt-4o-2024-05-13', system_content="", 
                          max_tokens=512, output_file=jsonl_file_path)

        batch_input_file = client.files.create(
            file=open(jsonl_file_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "synthetic_data_set_creation"
            }
        )
        print(f"The file {jsonl_file_path} has been created and sent to the OpenAI batch server")
    else:
        # Notify the user that the JSONL file already exists and the code won't run
        print(f"The file {jsonl_file_path} already exists. No action will be taken.")
    return batch_input_file, batch_input_file_id, jsonl_file_path


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Loading the batch generated data

        As the data is provided and it cannot be programmatically uploaded and then downloaded as you need the job number, This part of the notebook is not entirely reproducible without either manually downloading the results or loading the provided results.
        """
    )
    return


@app.cell
def __(all_prompts_df, convert_batch_to_dataframe):
    synthetic_data_df = convert_batch_to_dataframe('./data/test_vals.jsonl')#convert_batch_to_dataframe('./data/synthetic_articles.jsonl')
    #Clean up the text as some markdown was applied
    synthetic_data_df['generated_content'] = synthetic_data_df['generated_content'].str.replace(r"\*|#", "", regex=True)

    #merge back on the parameters used to generate the article
    synthetic_data_df = all_prompts_df.merge(synthetic_data_df, left_on = 'file_name', right_on =  'id')
    #Drop the file name but retain the identical ID
    synthetic_data_df.drop(columns='file_name', inplace=True)

    #Save to use as the base data
    synthetic_data_df.to_csv('./data/synthetic_articles.csv')
    return synthetic_data_df,


@app.cell
def __(synthetic_data_df):
    #The vast majority of data was in excess of the 500 token limit when generating the text which is to be expected as words are usually more than one token
    print(f"Total number of tokens in dataset {synthetic_data_df['completion_tokens'].sum()}")
    synthetic_data_df.groupby('finish_reason').size()
    return


@app.cell
def __(synthetic_data_df):
    synthetic_data_df.loc[2,'generated_content']
    return


@app.cell
def __(random):
    def get_random_token_window(df, target_tokens, text_column, tokenizer, seed = 1812):
        """
        Returns a DataFrame with an additional column containing a randomly selected
        token window from the specified text column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_tokens (int): The target number of tokens to select.
            text_column (str): The name of the column containing the text.

        Returns:
            pd.DataFrame: The DataFrame with an additional column 'token_window'.
        """
        # Set the random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        def select_token_window(text):
            # Tokenize the text
            tokens = tokenizer.encode(text)
            num_tokens = len(tokens)

            # If the number of tokens is less than or equal to target_tokens, return the whole text
            if num_tokens <= target_tokens:
                return text

            # Randomly select a start point within the tokenized text
            start_idx = random.randint(0, num_tokens - target_tokens)
            selected_tokens = tokens[start_idx:start_idx + target_tokens]

            # Decode the selected tokens back into text
            selected_text = tokenizer.decode(selected_tokens)

            return selected_text

        # Apply the function to the specified text column
        df['token_window'] = df[text_column].apply(select_token_window)

        return df
    return get_random_token_window,


@app.cell
def __(get_random_token_window, synthetic_data_df, tokenizer):
    test = get_random_token_window(synthetic_data_df, target_tokens = 200, text_column = 'generated_content', tokenizer = tokenizer)
    return test,


@app.cell
def __(test):
    test.loc[1, 'token_window']
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Save the dataset

        The dataset has now been created and can be save for use in other parts of the project
        """
    )
    return


@app.cell
def __(test):
    test.drop(columns = 'generated_content').to_parquet('./data/subset_synth_data.parquet')
    return


if __name__ == "__main__":
    app.run()
