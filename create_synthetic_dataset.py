import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


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
    from synthetic_data_functions import (
        get_from_wikipedia,
        process_wiki_time_line,
        process_wiki_timeline_format2,
        generate_prompts,
        split_generated_content,
        get_random_token_window,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    import math

    # import math
    from openai import OpenAI

    client = OpenAI()
    load_dotenv()
    wiki_user_agent = os.getenv("wiki_user_agent")
    return (
        AutoTokenizer,
        OpenAI,
        client,
        generate_prompts,
        get_from_wikipedia,
        get_random_token_window,
        json,
        load_dotenv,
        math,
        np,
        openai,
        os,
        pd,
        process_wiki_time_line,
        process_wiki_timeline_format2,
        random,
        re,
        split_generated_content,
        sys,
        tiktoken,
        tokenizer,
        tqdm,
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
    c19_timeline = get_from_wikipedia(
        wiki_user_agent, "Timeline_of_the_19th_century", language="en"
    )

    c19_timeline = process_wiki_time_line(c19_timeline)

    c19_timeline = c19_timeline.iloc[
        : c19_timeline[
            c19_timeline["content"].str.contains("References", case=True)
        ].index[0],
        :,
    ]
    return (c19_timeline,)


@app.cell
def __(c19_timeline):
    c19_timeline
    return


@app.cell
def __(get_from_wikipedia, process_wiki_timeline_format2, wiki_user_agent):
    brit_diplo_timeline = get_from_wikipedia(
        wiki_user_agent, "Timeline_of_British_diplomatic_history", language="en"
    )
    brit_diplo_timeline = process_wiki_timeline_format2(brit_diplo_timeline)

    # remove the bibliography and so as this is not part of the timeline
    brit_diplo_timeline = brit_diplo_timeline.iloc[
        : brit_diplo_timeline[
            brit_diplo_timeline["content"].str.contains("biblio", case=False, na=False)
        ].index[0],
        :,
    ]
    # subset timeline to only the 19th century
    brit_diplo_timeline = brit_diplo_timeline.loc[
        (brit_diplo_timeline["year"] > 1799) & (brit_diplo_timeline["year"] < 1900)
    ]
    return (brit_diplo_timeline,)


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
        "personal diary entry",
    ]

    writing_style = [
        "formal",
        "informal",
        "satirical",
        "religious",
        "polemic",
        "romantic",
        "persuasive",
        "descriptive",
    ]

    #'radical' can't be used instead of chartist as it doesn't seem to generate named people as frequently
    persona = [
        "general public",
        "women's rights",
        "politics",
        "economics and trade",
        "military",
        "reactionary",
        "chartist",
        "clergy",
        "arts and culture",
    ]

    sentiment = ["positive", "neutral", "negative"]

    complexity = [
        "simple",
        "moderate",
        "advanced",
    ]
    writing_style_combinations = (
        len(text_type)
        * len(writing_style)
        * len(persona)
        * len(sentiment)
        * len(complexity)
    )
    print(f"Total number of writing style combinations {writing_style_combinations}")
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
    all_prompts_df = generate_prompts(
        pd.concat([c19_timeline, brit_diplo_timeline], ignore_index=True),
        text_type,
        writing_style,
        persona,
        sentiment,
        complexity,
        num_samples=11000,
        word_count=300,
        seed=1865,
    )

    all_prompts_df["file_name"] = "index_" + all_prompts_df.index.astype(str)

    all_prompts_df.loc[11, "full_prompt"]
    return (all_prompts_df,)


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
        with open(output_file, "w") as file:
            for _, row in df.iterrows():
                messages = [{"role": "system", "content": system_content}]

                messages.append({"role": "user", "content": row["full_prompt"]})

                entry = {
                    "custom_id": f"{row['file_name']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                    },
                }
                # Write each JSON object as a separate line in the JSONL file
                file.write(json.dumps(entry) + "\n")

    def convert_batch_to_dataframe(jsonl_path):
        """
        Extract 'custom_id', 'assistant' 'content', and 'usage' from a JSON string and create a DataFrame.

        Parameters:
        dict_list (str): The input a list of dictionaries.

        Returns:
        pd.DataFrame: DataFrame containing the extracted data.
        """
        with open(jsonl_path, "r") as file:
            dict_list = [json.loads(line) for line in file]

        extracted_data = []

        for json_object in dict_list:
            data = json_object
            custom_id = data["custom_id"]
            assistant_content = data["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            finish_reason = data["response"]["body"]["choices"][0]["finish_reason"]
            usage = data["response"]["body"]["usage"]

            row_data = {
                "id": custom_id,
                "generated_content": assistant_content,
                "finish_reason": finish_reason,
            }

            # Add the usage dictionary to the row data
            row_data.update(usage)

            extracted_data.append(row_data)

        return pd.DataFrame(extracted_data)

    return convert_batch_to_dataframe, create_jsonl_file


@app.cell
def __(all_prompts_df, client, create_jsonl_file, os):
    # Define the path to the JSONL file
    jsonl_file_path = "./data/for_gpt.jsonl"

    # Check if the JSONL file exists
    if not os.path.exists(jsonl_file_path):
        # If the file does not exist, create it and proceed with the batch request
        create_jsonl_file(
            all_prompts_df,
            model="gpt-4o-2024-05-13",
            system_content="",
            max_tokens=512,
            output_file=jsonl_file_path,
        )

        batch_input_file = client.files.create(
            file=open(jsonl_file_path, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "synthetic_data_set_creation"},
        )
        print(
            f"The file {jsonl_file_path} has been created and sent to the OpenAI batch server"
        )
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
    synthetic_data_df = convert_batch_to_dataframe("./data/synthetic_articles.jsonl")
    # Clean up the text as some markdown was applied
    synthetic_data_df["generated_content"] = synthetic_data_df[
        "generated_content"
    ].str.replace(r"\*|#", "", regex=True)

    # merge back on the parameters used to generate the article
    synthetic_data_df = all_prompts_df.merge(
        synthetic_data_df, left_on="file_name", right_on="id"
    )
    # Drop the file name but retain the identical ID
    synthetic_data_df.drop(columns="file_name", inplace=True)

    # Save to use as the base data
    synthetic_data_df.to_csv("./data/synthetic_articles.csv")
    return (synthetic_data_df,)


@app.cell
def __(synthetic_data_df):
    print(
        f"Total number of tokens in dataset {synthetic_data_df['completion_tokens'].sum()}"
    )
    synthetic_data_df.groupby("finish_reason").size()
    return


@app.cell
def __(synthetic_data_df):
    synthetic_data_df.loc[
        synthetic_data_df["generated_content"].str.contains(
            "Railway, heralded as the world's first public railway, has made its grand inauguration in our blessed England. Be it known, dear diary, that "
        ),
        :"generated_content",
    ]
    return


@app.cell
def __(synthetic_data_df):
    synthetic_data_df.loc[6:9, :"generated_content"]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Create the synthetic datasets

        These datasets contain different lengths of text and observations but the same total number of tokens
        """
    )
    return


@app.cell
def __(
    get_random_token_window,
    os,
    split_generated_content,
    synthetic_data_df,
    tokenizer,
):
    # Define the target tokens and corresponding number of splits
    target_tokens_list = [200, 100, 50, 25, 10]
    num_splits_list = [1, 2, 4, 8, 20]

    # Loop through each pair of target tokens and num_splits
    for target_tokens, num_splits in zip(target_tokens_list, num_splits_list):
        output_path = f"./data/synth_gt/synth{target_tokens}.parquet"

        # Check if the file already exists
        if not os.path.exists(output_path):
            # Split the content
            synth_df = split_generated_content(
                synthetic_data_df,
                id_col="id",
                content_col="generated_content",
                num_splits=num_splits,
            )

            # Get random token window
            synth_df = get_random_token_window(
                synth_df,
                target_tokens=target_tokens,
                text_column="generated_content",
                tokenizer=tokenizer,
            )

            # Select and rename columns
            synth_df = synth_df[["id", "sub_id", "token_window"]].copy()
            synth_df.rename(columns={"token_window": "gt_text"}, inplace=True)

            synth_df["data_type"] = (
                ["training"] * 10000 * num_splits
                + ["validation"] * 500 * num_splits
                + ["test"] * 500 * num_splits
            )

            # Save to parquet
            synth_df.to_parquet(output_path)
            print(f"File created: {output_path}")
        else:
            print(f"File already exists: {output_path}")
    return (
        num_splits,
        num_splits_list,
        output_path,
        synth_df,
        target_tokens,
        target_tokens_list,
    )


@app.cell
def __(pd):
    orig_obs_list = [8192, 4096, 2048, 1024, 512, 256, 128]

    total_tokens = [200 * orig_obs for orig_obs in orig_obs_list]

    token_per_obs = [200, 100, 50, 25, 10]

    # Modify the DataFrame as per the updated instructions

    # Create the dataframe with total_tokens as index
    df_latex = pd.DataFrame({"total_tokens": total_tokens})

    # Add the token_per_obs columns
    for token in token_per_obs:
        df_latex[token] = (df_latex["total_tokens"] / token).astype(int)

    # Set total_tokens as the index
    df_latex.set_index("total_tokens", inplace=True)

    # Rename columns and create a multi-level column for "Tokens per text"
    df_latex.columns = pd.MultiIndex.from_product(
        [["Tokens per text"], df_latex.columns]
    )

    # Convert the dataframe to LaTeX
    latex_output = df_latex.to_latex()

    latex_output
    return (
        df_latex,
        latex_output,
        orig_obs_list,
        token,
        token_per_obs,
        total_tokens,
    )


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
def __(pd):
    example_data = pd.read_parquet("./data/synth_gt/synth200.parquet")

    example_data
    return (example_data,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Training datasets for model comparison

        This section creates training datasets using the BLN600, CA and SMH datasets

        The data is split such that the median gt observation is 200 tokens as this is the number of tokens gave the best results, and also the datasets are quite small so more tokens would make the number of observations very low
        """
    )
    return


@app.cell
def __(os, pd):
    def process_folders(corrected_folder, uncorrected_folder, n=5):
        data = []

        # Get list of files in corrected folder
        files = os.listdir(corrected_folder)

        for file in files:
            corrected_path = os.path.join(corrected_folder, file)
            uncorrected_path = os.path.join(uncorrected_folder, file)

            # Read files
            with open(corrected_path, "r") as f:
                corrected_text = f.read().splitlines()
            with open(uncorrected_path, "r") as f:
                uncorrected_text = f.read().splitlines()

            # Split texts into chunks of n lines
            for i in range(0, len(corrected_text), n):
                corrected_chunk = "\n".join(corrected_text[i : i + n])
                uncorrected_chunk = "\n".join(uncorrected_text[i : i + n])

                # Calculate split number
                split_num = i // n + 1

                # Append to data list
                data.append(
                    {
                        "file_name": file,
                        "split_number": split_num,
                        "gt_text": corrected_chunk,
                        "ocr_text": uncorrected_chunk,
                    }
                )

        df = pd.DataFrame(data)
        df["gt_text"] = df["gt_text"].str.replace("\n", " ")
        df["ocr_text"] = df["ocr_text"].str.replace("\n", " ")

        return df

    return (process_folders,)


@app.cell
def __(process_folders, tokenizer):
    SMH_data = process_folders(
        corrected_folder="data/overproof/SMH/corrected",
        uncorrected_folder="data/overproof/SMH/uncorrected",
        n=23,
    )

    SMH_data["gt_tokens"] = SMH_data["gt_text"].apply(
        lambda x: len(tokenizer.encode(x))
    )
    SMH_data["ocr_tokens"] = SMH_data["ocr_text"].apply(
        lambda x: len(tokenizer.encode(x))
    )

    SMH_data.to_parquet("data/compare_datasets/SMH.parquet")

    print(SMH_data[["gt_tokens", "ocr_tokens"]].median())

    print(SMH_data[["gt_tokens", "ocr_tokens"]].sum())

    return (SMH_data,)


@app.cell
def __(process_folders, tokenizer):
    CA_data = process_folders(
        corrected_folder="data/overproof/CA/corrected",
        uncorrected_folder="data/overproof/CA/uncorrected",
        n=31,
    )

    CA_data["gt_tokens"] = CA_data["gt_text"].apply(lambda x: len(tokenizer.encode(x)))
    CA_data["ocr_tokens"] = CA_data["ocr_text"].apply(
        lambda x: len(tokenizer.encode(x))
    )

    CA_data.to_parquet("data/compare_datasets/CA.parquet")

    print(CA_data[["gt_tokens", "ocr_tokens"]].median())

    print(CA_data[["gt_tokens", "ocr_tokens"]].sum())
    return (CA_data,)


@app.cell
def __(CA_data, SMH_data, pd):
    combined_data = pd.concat([CA_data, SMH_data], ignore_index=True)

    combined_data.to_parquet("data/compare_datasets/overproof.parquet")

    print(combined_data[["gt_tokens", "ocr_tokens"]].median())

    print(combined_data[["gt_tokens", "ocr_tokens"]].sum())

    return (combined_data,)


@app.cell
def __(pd, tokenizer):
    BLN600 = pd.read_csv("data/BLN600/ocr_paper_data/train.csv")

    BLN600.rename(
        columns={"OCR Text": "ocr_text", "Ground Truth": "gt_text"}, inplace=True
    )

    BLN600["gt_tokens"] = BLN600["gt_text"].apply(lambda x: len(tokenizer.encode(x)))
    BLN600["ocr_tokens"] = BLN600["ocr_text"].apply(lambda x: len(tokenizer.encode(x)))

    BLN600.to_parquet("data/compare_datasets/BLN600.parquet")
    print(BLN600[["gt_tokens", "ocr_tokens"]].median())

    print(BLN600[["gt_tokens", "ocr_tokens"]].sum())
    return (BLN600,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Create example of token length

        in order to give an example of what 200, 100, 50 25, 10 tokens looks like I will use one of the synthetic articles and highlight each text length in a different colour in Latex. This is shown in the supplementary material
        """
    )
    return


@app.cell
def __(pd, tokenizer):
    _temp = pd.read_parquet("./data/synth_gt/synth200.parquet")

    tokens = tokenizer.tokenize(_temp.loc[8, "gt_text"])

    # Split into segments
    first_100 = tokens[:100]  # First 100 tokens
    next_50 = tokens[100:150]  # Next 50 tokens
    next_25 = tokens[150:175]  # Next 25 tokens
    next_10 = tokens[175:185]  # Next 10 tokens

    # Combine back into text if needed
    first_100_text = tokenizer.convert_tokens_to_string(first_100)
    next_50_text = tokenizer.convert_tokens_to_string(next_50)
    next_25_text = tokenizer.convert_tokens_to_string(next_25)
    next_10_text = tokenizer.convert_tokens_to_string(next_10)
    return (
        first_100,
        first_100_text,
        next_10,
        next_10_text,
        next_25,
        next_25_text,
        next_50,
        next_50_text,
        tokens,
    )


@app.cell
def __(first_100_text):
    first_100_text
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
