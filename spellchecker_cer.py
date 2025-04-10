import marimo

__generated_with = "0.8.18"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Reviewer requested spell check score

        One reviewer requested a comparison with using a simple spell check. This script does that.
        """
    )
    return


@app.cell
def __():
    import os
    import pandas as pd
    from textblob import TextBlob


    def load_text_files(folder_path):
        file_names = []
        texts = []
        
        # Get all txt files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    file_names.append(file_name)
                    texts.append(text)
        
        # Create DataFrame
        df = pd.DataFrame({
            'file_name': file_names,
            'gt_text': texts
        })
        
        return df
    #load and spellcheck files, used to get load up the corrupted files
    def spell_check_files(folder_path):
        # First load all files into DataFrame
        df = load_text_files(folder_path)
        
        # Add spell-checked text as new column
        df['ocr_text'] = df['gt_text'].apply(lambda x: str(TextBlob(x).correct()))
        
        return df[['file_name', 'ocr_text']]  # Return only requested columns

    return TextBlob, load_text_files, os, pd, spell_check_files


@app.cell
def __(load_text_files, spell_check_files):

    # Specify your folder path here
    ocr_path = "data/ncse/transcription_raw_ocr"
    gt_path = "data/ncse/transcription_files"

    # Process files and get results
    results = spell_check_files(ocr_path).merge(load_text_files(gt_path), on = 'file_name')

    # Display results
    print("\nResults DataFrame:")
    print(results)

    # Optionally save to CSV
    #results.to_csv('spell_check_results.csv', index=False)
    return gt_path, ocr_path, results


@app.cell
def __(results):
    import evaluate
    from lm_support_functions import compute_metric

    metric_cer = evaluate.load("cer")
    metric_wer = evaluate.load("wer")

    temp = results.copy()

    temp['cer'] = temp.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='ocr_text', reference_col='gt_text')
    temp['wer'] = temp.apply(compute_metric, axis=1, metric =metric_wer, prediction_col='ocr_text', reference_col='gt_text')
    return compute_metric, evaluate, metric_cer, metric_wer, temp


@app.cell
def __(temp):
    temp[['cer', 'wer']].median()
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
