
import pandas as pd
import re
import wikipediaapi
import random


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


def create_prompt_for_writing(text, text_type, writing_style, audience, sentiment, complexity, year, word_count = 500):
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

    text = f"""It is the year {year}. Using the text provided below surrounded by triple #, write a {word_count} word {text_type}, from a {writing_style} perspective, intended for {audience}, with a {sentiment} tone and {complexity} language
    ###
    {text}
    ###""" 

    return text


def generate_prompts(df_content, text_type, writing_style, audience, sentiment, complexity, num_samples=10, word_count=500):
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

    # Step 1: Generate random combinations
    prompt_df = [
        {
            "text_type": random.choice(text_type),
            "writing_style": random.choice(writing_style),
            "audience": random.choice(audience),
            "sentiment": random.choice(sentiment),
            "complexity": random.choice(complexity)
        }
        for _ in range(num_samples)
    ]

    # Step 2: Create a DataFrame from the list of dictionaries
    df_random = pd.DataFrame(prompt_df)
    
    # Step 3: Randomly sample the content and year columns together to maintain pairing
    sampled_df = df_content.sample(n=num_samples, replace=True).reset_index(drop=True)
    df_random['content'] = sampled_df['content']
    df_random['year'] = sampled_df['year']

    # Step 4: Create the full prompt using the create_prompt_for_writing function
    df_random['full_prompt'] = df_random.apply(
        lambda row: create_prompt_for_writing(
            row['content'], 
            row['text_type'], 
            row['writing_style'], 
            row['audience'], 
            row['sentiment'], 
            row['complexity'],
            row['year'],
            word_count=word_count
        ), axis=1
    )

    return df_random
