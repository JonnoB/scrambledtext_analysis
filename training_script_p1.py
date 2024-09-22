from scrambledtext import (
    ProbabilityDistributions,
    CorruptionEngine,
    WERBasedCorruptionEngine,
)

import pandas as pd
import argparse
import ast
import os
from datasets import Dataset, DatasetDict, load_from_disk
from lm_support_functions import training_prompt, compute_metric, infer_on_test_set
from unsloth import FastLanguageModel
import torch
from trl import (
    SFTTrainer,
)  # , SFTConfig #the confilarge_gold_datag does not exist in the trl required by unsloth
from transformers import TrainingArguments
import wandb

wandb.login()  # you need to have your wandb credentials stored
import time
import evaluate


# Initialize the parser
parser = argparse.ArgumentParser(description="Process corruption type and arguments.")

# Add arguments
parser.add_argument("corruption_type", type=str, help="Type of corruption to apply")
parser.add_argument(
    "corruption_args",
    type=str,
    help="Arguments for the corruption as a dictionary-like string",
)
parser.add_argument(
    "dataset", type=str, help="path to the dataset stored as a parquet file"
)
parser.add_argument("output", type=str, help="path to the output folder")

# Parse the arguments
args = parser.parse_args()


# Access the arguments
corruption_type = args.corruption_type

# Convert corruption_args to a dictionary
corruption_args = ast.literal_eval(args.corruption_args)

dataset_path = args.dataset

# all experiments in the project will be saved here
output_path = args.output


results_path = os.path.join(output_path, "results")

# create the output folder if it doesn't already exist
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(results_path)

# get the dataset file name which will be used as part of the output filename for the results
file_name = os.path.splitext(os.path.basename(dataset_path))[0]

synth_data = pd.read_parquet(dataset_path)


# For now use a fixed corruption
print("Create corruption tables")
corruption_data = pd.read_csv("./data/aligned/aligned_BLN600.csv")

gt_aligned_list = corruption_data["gt_aligned"].to_list()

noise_aligned_list = corruption_data["noise_aligned"].to_list()

aligned_texts = list(zip(gt_aligned_list, noise_aligned_list))

corruption_probs = ProbabilityDistributions(aligned_texts)


#
# Corrupting the text
# The below if statements allow the text to be corrupted dependent on the arguments provided to the script
# The idea is to make it easy to run different experiments with the same script
#
print("creating corrupted text")
if corruption_type == "cer":
    # reset the cer based on the arguments
    corruption_probs.modify_and_renormalize_probs(
        column="correct", desired_value=corruption_args["cer"], inplace=True
    )

    corruption_function = CorruptionEngine(
        corruption_probs.conditional,
        corruption_probs.substitutions,
        corruption_probs.insertions,
    )

    synth_data["ocr_text"], synth_data["cer"] = zip(
        *synth_data["gt_text"].apply(
            lambda text: corruption_function.corrupt_text(text)
        )
    )

    output_file_name = f"""{file_name}_{int(corruption_args['cer']*100)}.csv"""

elif corruption_type == "cer_wer":
    corruption_function = WERBasedCorruptionEngine(
        corruption_probs.conditional,
        corruption_probs.substitutions,
        corruption_probs.insertions,
    )

    synth_data["ocr_text"], synth_data["wer"], synth_data["cer"] = zip(
        *synth_data["gt_text"].apply(
            lambda text: corruption_function.corrupt_text_with_wer_cer(
                text,
                target_wer=corruption_args["wer"],
                target_cer=corruption_args["cer"],
            )
        )
    )

    output_file_name = f"""{file_name}_{int(corruption_args['cer']*100)}_{int(corruption_args['wer']*100)}.csv"""
else:
    print("No correct argument entered hot crash incoming")


hf_dataset = Dataset.from_pandas(synth_data)

# Split the dataset based on the 'data_type' column into training, validation, and test sets
dataset_dict = DatasetDict(
    {
        "train": hf_dataset.filter(lambda example: example["data_type"] == "training"),
        "validation": hf_dataset.filter(
            lambda example: example["data_type"] == "validation"
        ),
        "test": hf_dataset.filter(lambda example: example["data_type"] == "test"),
    }
)

# clean up uneccessary dataframes
del hf_dataset
del synth_data


##
## Load model and set parameters
##

max_seq_length = 768  # 512

# We take the instruct model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  #'unsloth/llama-3-8b-bnb-4bit',#'unsloth/Phi-3-mini-4k-instruct-bnb-4bit',#"unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,
)

# Sometimes you may get an "offload to cpu" type error, this can happen if you stop/crash part way through training, #
# check the VRAM on the GPU is not full of the old model


model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)


dataset_dict = dataset_dict.map(
    lambda x: training_prompt(x, "ocr_text", "gt_text", tokenizer), batched=False
)


args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    max_steps=-1,  # should be -1
    num_train_epochs=2,
    learning_rate=5e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=16,
    eval_steps=128,
    evaluation_strategy="steps",  # For this version of trl need to use this setting. This changes for more recent versions I think
    save_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    remove_unused_columns=True,
    seed=3407,
    output_dir="outputs",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_dict["train"],  # NO EVAL FOR THIS SET OF EXPERIMENTS
    dataset_text_field="full_prompt",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,  # Can make training 5x faster for short sequences.
    args=args,
)


##
## Begin Training
##


run = wandb.init(
    # set the wandb project where this run will be logged
    project=os.path.basename(output_path)
)

# Get the W&B run name
run_name = run.name

# Training
start = time.time()
trainer.train()
print(f"Training complete: {time.time() - start}")


##
## Post training
##


##
## Load ncse dataset
##

data = load_from_disk("ncse_hf_dataset")

##
## infer over test set
##

# switch to inference mode
FastLanguageModel.for_inference(model)

temp = infer_on_test_set(data, model, tokenizer)

## These multiple saves are to make sure there isn't an error losing everything if for example the LLM makes an error
temp.to_csv(output_file_name)

temp["clocrc_text"] = temp["clocrc_text"].apply(
    lambda x: x.split("###Recovery###")[1].split("###")[0]
)

temp.to_csv(output_file_name)


##
## add wer and cer
##

metric_cer = evaluate.load("cer")
metric_wer = evaluate.load("wer")


# Apply the function to each row for 'output' and 'raw_text' columns
temp["type"] = "llama 3.1 base"
temp["tokens"] = temp["article_text"].apply(lambda x: len(tokenizer.encode(x)))
temp["cer"] = temp.apply(
    compute_metric,
    axis=1,
    metric=metric_cer,
    prediction_col="clocrc_text",
    reference_col="gt_text",
)
temp["wer"] = temp.apply(
    compute_metric,
    axis=1,
    metric=metric_wer,
    prediction_col="clocrc_text",
    reference_col="gt_text",
)


# Compute the ERP (Error Reduction Percentage)
temp["erp_cer"] = (temp["wer_orig"] - temp["wer"]) / temp["wer_orig"]
temp["erp_wer"] = (temp["wer_orig"] - temp["wer"]) / temp["wer_orig"]

temp.to_csv(output_file_name)


print("Train and test completed terminating script successfully")
