import marimo

__generated_with = "0.7.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    return np, os, pd, plt, sns


@app.cell
def __(mo):
    mo.md(
        r"""
        # Analysing Model performance

        This notebook is to model the performance of the LLM's that have been trained to try and visual how well they are working and find strategies that do better
        """
    )
    return


@app.cell
def __(np):
    def wer_cer_loss(x, y, x_ref1, y_ref1, x_ref2, y_ref2):
        # Normalize the coordinates
        x_norm = (x - x_ref1) / (x_ref2 - x_ref1)
        y_norm = (y - y_ref1) / (y_ref2 - y_ref1)

        # Set negative normalized values to zero
        x_norm_star = max(0, x_norm)
        y_norm_star = max(0, y_norm)

        # Calculate the loss
        loss = np.sqrt(x_norm_star**2 + y_norm_star**2)

        return loss
    return wer_cer_loss,


@app.cell
def __(os, pd, wer_cer_loss):
    model_performance_df = [pd.DataFrame({'model':"GPT4", "wer":0.17, "cer":0.09}, index = [0])]

    #get the original wer and cer using wer_orig and cer_orig
    temp = pd.read_csv(os.path.join('data/results','ncse_test_recovered_base_llama.csv'))
    temp = temp[['wer_orig', 'cer_orig']].median().to_frame().T
    temp.rename(columns = {'wer_orig':'wer', 'cer_orig':'cer'}, inplace = True)
    temp['model'] = 'original'

    model_performance_df.append(temp)

    #add base llama
    temp = pd.read_csv(os.path.join('data/results','ncse_test_recovered_base_llama.csv'))
    temp = temp[['wer', 'cer']].median().to_frame().T
    temp['model'] = 'base llama'

    model_performance_df.append(temp)

    _folder_path = 'data/cer_exp/results'
    for _file in os.listdir(_folder_path):

        _temp = pd.read_csv(os.path.join(_folder_path, _file))
        _temp = _temp[['wer', 'cer']].median().to_frame().T
        _temp['model'] = _file.replace(".csv", "")

        model_performance_df.append(_temp)

    model_performance_df = pd.concat(model_performance_df, ignore_index=True).sort_values('cer')

    model_performance_df['total_error'] = model_performance_df.apply(lambda row: 
                                                                     wer_cer_loss(row['cer'], row['cer'], 0.17, 0.09, 0.41, 0.304 ), axis = 1)
    return model_performance_df, temp


@app.cell
def __(model_performance_df):
    model_performance_df.sort_values('cer')
    return


@app.cell
def __(mo):
    mo.md(r"""# Creating a plot that visusalises the change in CER and WER for different models and approaches""")
    return


@app.cell
def __(model_performance_df, plt, sns):
    sns.scatterplot(data = model_performance_df, x = 'cer', y = 'wer')
    plt.title('CER and WER performance relative to\nGPT4 and LLama 3.1 base instruct')

    # Get the cer and wer values for 'GPT4'
    gpt4_values = model_performance_df[model_performance_df['model'] == 'GPT4']
    gpt4_cer = gpt4_values['cer'].values[0]
    gpt4_wer = gpt4_values['wer'].values[0]

    # Get the cer and wer values for 'base'
    base_values = model_performance_df[model_performance_df['model'] == 'base llama']
    base_cer = base_values['cer'].values[0]
    base_wer = base_values['wer'].values[0]

    # Add infinite vertical and horizontal lines for 'GPT4'
    plt.axvline(x=gpt4_cer, color='blue', linestyle='-', label='GPT4')
    plt.axhline(y=gpt4_wer, color='blue', linestyle='-')

    # Add infinite vertical and horizontal lines for 'base'
    plt.axvline(x=base_cer, color='red', linestyle='-', label='Base Llama')
    plt.axhline(y=base_wer, color='red', linestyle='-')


    #Create the frame relative to perfect performance
    plt.xlim([0, 0.55])
    plt.ylim([0, 0.65])
    plt.legend()
    plt.show()
    return base_cer, base_values, base_wer, gpt4_cer, gpt4_values, gpt4_wer


@app.cell
def __(os, pd, sns):
    cer_vals_df = []

    _folder_path = 'data/cer_exp/results'
    for _file in os.listdir(_folder_path):

        _temp = pd.read_csv(os.path.join(_folder_path, _file))
        _temp = _temp[['wer', 'cer']].median().to_frame().T
        _temp['model'] = _file.replace(".csv", "")

        cer_vals_df.append(_temp)

    cer_vals_df = pd.concat(cer_vals_df, ignore_index=True)


    cer_vals_df['target_cer'] = 100 - cer_vals_df['model'].str.replace('synth200_', "").astype(int)

    sns.lineplot(data = cer_vals_df, x = 'target_cer', y = 'wer')
    sns.lineplot(data = cer_vals_df, x = 'target_cer', y = 'cer')
    return cer_vals_df,


@app.cell
def __(os, pd):
    temp2 = pd.read_csv(os.path.join("data/cer_exp/results/synth200_80.csv"))
    return temp2,


@app.cell
def __(temp2):
    temp2
    return


if __name__ == "__main__":
    app.run()
