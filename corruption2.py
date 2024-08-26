import marimo

__generated_with = "0.7.20"
app = marimo.App(width="medium")


@app.cell
def __():
    #python training_script.py cer '{"cer":0.2}' synth_gt/synth200.parquet cer_exp

    from scrambledtext import (ProbabilityDistributions, CorruptionEngine, modify_and_renormalize_probs)

    import pandas as pd
    import numpy as np

    import time 
    import evaluate
    return (
        CorruptionEngine,
        ProbabilityDistributions,
        evaluate,
        modify_and_renormalize_probs,
        np,
        pd,
        time,
    )


@app.cell
def __(CorruptionEngine, ProbabilityDistributions, pd):
    ##
    ## Testing only three being used
    ##
    synth_data = pd.read_parquet('data/synth_gt/synth200.parquet')#.sample(3)


    #For now use a fixed corruption
    print('Create corruption tables')

    corruption_probs = ProbabilityDistributions()
    #load the premade corruption distribution
    corruption_probs = corruption_probs.load_from_json('data/learned_corruption_distribs.json')


    #
    # Corrupting the text
    # The below if statements allow the text to be corrupted dependent on the arguments provided to the script
    # The idea is to make it easy to run different experiments with the same script
    #

    #reset the cer based on the arguments, the cer is remember the probability of correct = 1-CER
    #corruption_probs.modify_and_renormalize_probs(column='correct', desired_value=1-corruption_args['cer'], inplace=True)

    corruption_function = CorruptionEngine(corruption_probs.conditional, corruption_probs.substitutions, corruption_probs.insertions, target_cer= 0.2, target_wer=0.6)

    synth_data['ocr_text'], synth_data['wer'], synth_data['cer'], synth_data['effect_cer'] = zip( *synth_data['gt_text'].apply(lambda text:corruption_function.corrupt_text(text)))
    return corruption_function, corruption_probs, synth_data


@app.cell
def __(synth_data):
    synth_data
    return


if __name__ == "__main__":
    app.run()
