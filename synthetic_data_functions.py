
import pandas as pd
import re
import wikipediaapi
import random
import math

def get_from_wikipedia(user_agent, page, language = 'en'):
    """ 
    Calls the wikipedia server and attempts to return the text from the specified page.
    """
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, language)

    page = wiki_wiki.page(page)

    if page.exists():
        content = page.text

    else:
        print("Page does not exist")

    return content

def process_wiki_time_line(text):
    # identifies year split given that is is preceeded and followed by a line break
    pattern = r"\n(\d{4})\n"

    # Adding a line break at the start and end of the text to ensure proper splitting
    text = "\n" + text.strip() + "\n"

    # Splitting the text by the year pattern
    split_text = re.split(pattern, text)

    # Organizing the data into a list of dictionaries
    data = []
    for i in range(1, len(split_text), 2):
        year = split_text[i].strip()
        content = split_text[i+1].strip()
        events = content.splitlines()
        for event in events:
            if event:  # Avoid empty lines
                data.append({"year": int(year), "content": event})

    return pd.DataFrame(data)

def process_wiki_timeline_format2(text):

    pattern = r"(\d{4})(?:–\d{2,4})?:"

    # Splitting the text based on the pattern
    split_text = re.split(pattern, text)

    # Organizing the data into a list of dictionaries
    data = []
    for i in range(1, len(split_text), 2):
        year = split_text[i].strip()[:4]  # Taking the first year in the range
        content = split_text[i+1].strip()
        events = content.splitlines()
        for event in events:
            if event:  # Avoid empty lines
                data.append({"year": int(year), "content": event})

    # Creating a DataFrame
    df = pd.DataFrame(data)

    return df

def create_prompt_for_writing(text, text_type, writing_style, persona, sentiment, complexity, year, word_count = 500):
    """
    Generate a writing prompt based on the provided text and parameters.

    This function creates a structured writing prompt by filling in a template
    with specified parameters such as text type, writing style, intended audience,
    sentiment, and complexity. The generated prompt is intended for producing 
    contemporary writing based on the provided text.

    Args:
        text (str): The content to be included in the prompt, which will be 
            surrounded by triple # symbols.
        text_type (str): The type of text to be written (e.g., newspaper article,
            obituary, editorial).
        writing_style (str): The style in which the text should be written 
            (e.g., formal, informal, satirical).
        audience (str): The intended audience for the writing (e.g., general public,
            scholars, children).
        sentiment (str): The tone or sentiment of the writing (e.g., somber,
            optimistic, critical).
        complexity (str): The complexity level of the language to be used (e.g.,
            simple, advanced, dense).
        year (int): The year in which the work is written
        word_count (int, optional): The desired word count for the writing. Defaults
            to 500.

    Returns:
        str: A formatted writing prompt with the provided parameters.
    """
#added in comment on offensiveness as after reading the "negative" comments on twelve years a slave which were all anti'slavery instead of anti-twelve years a slave aka pro-slavery. It is important that such text is generated as those were actual views back then
    text = f"""It is the year {year}. Using the text provided below surrounded by triple #, write a {word_count} word {writing_style} {text_type} with a {sentiment} sentiment, the persona of the writer is {persona}, the reading level should be {complexity}.
    Note: The resultant text may be distateful to modern readers that is ok. Respond only in plain text do not use markdown
    ###
    {text}
    ###""" 

    return text

def generate_prompts(df_content, text_type, writing_style, persona, sentiment, complexity, num_samples=10, word_count=500, seed = 1):
    """
    Generate a DataFrame with randomly sampled prompts based on provided categories,
    repeat the content column to match the length of the random combinations, and
    create a new column containing the full prompt.

    Args:
        df_content (pd.DataFrame): DataFrame containing the 'content' column to be repeated.
        text_type (list): List of possible text types (e.g., newspaper article, obituary).
        writing_style (list): List of possible writing styles (e.g., formal, satirical).
        audience (list): List of possible audiences (e.g., general public, scholars).
        sentiment (list): List of possible sentiments (e.g., somber, optimistic).
        complexity (list): List of possible language complexities (e.g., simple, advanced).
        num_samples (int, optional): Number of random samples to generate. Defaults to 10.
        word_count (int, optional): Word count for the generated prompt. Defaults to 500.

    Returns:
        pd.DataFrame: DataFrame containing the full prompts and corresponding attributes.
    """
    random.seed(seed)
    # Step 1: Generate random combinations
    prompt_df = [
        {
            "text_type": random.choice(text_type),
            "writing_style": random.choice(writing_style),
            "persona": random.choice(persona),
            "sentiment": random.choice(sentiment),
            "complexity": random.choice(complexity)
        }
        for _ in range(num_samples)
    ]

    # Step 2: Create a DataFrame from the list of dictionaries
    df_random = pd.DataFrame(prompt_df)
    
    # Step 3: Randomly sample the content and year columns together to maintain pairing
    sampled_df = df_content.sample(n=num_samples, replace=True, random_state=seed).reset_index(drop=True)
    df_random['content'] = sampled_df['content']
    df_random['year'] = sampled_df['year']

    # Step 4: Create the full prompt using the create_prompt_for_writing function
    df_random['full_prompt'] = df_random.apply(
        lambda row: create_prompt_for_writing(
            row['content'], 
            row['text_type'], 
            row['writing_style'], 
            row['persona'], 
            row['sentiment'], 
            row['complexity'],
            row['year'],
            word_count=word_count
        ), axis=1
    )

    return df_random



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


def split_generated_content(df, id_col, content_col, num_splits):
    """
    Splits the 'generated_content' in each row of the DataFrame into 'num_splits' equal parts,
    and expands these parts into multiple rows while retaining the 'id'.

    Parameters:
    - df: DataFrame containing the data.
    - id_col: The name of the column containing the ID.
    - content_col: The name of the column containing the generated content to split.
    - num_splits: Number of equal parts to split the content into.

    Returns:
    - A new DataFrame with each split content in its own row, along with an additional 'sub_id' column.
    """
    # Initialize an empty list to store the results
    result = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        content = row[content_col]
        content_length = len(content)

        # Calculate the size of each split
        split_size = math.ceil(content_length / num_splits)

        # Handle case where content is shorter than split size
        if content_length <= split_size:
            split_contents = [content]
        else:
            split_contents = [content[i:i + split_size] for i in range(0, content_length, split_size)]

        # Ensure exactly `num_splits` parts by padding with empty strings if needed
        split_contents.extend([''] * (num_splits - len(split_contents)))

        # Append each split part as a new row in the result list, with a sub_id
        for sub_id, part in enumerate(split_contents, start=1):
            result.append({id_col: row[id_col], content_col: part, 'sub_id': sub_id})

    # Convert the result list into a DataFrame
    result_df = pd.DataFrame(result)

    return result_df
