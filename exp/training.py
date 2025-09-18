import os
import json
import random
import pandas as pd
random.seed(42)

DATA_DIR = "/home/aiscuser/yangwang/data"
# Load the customer support data
df = pd.read_csv(os.path.join(DATA_DIR, "aa_dataset-tickets-multi-lang-5-2-50-version.csv"))
# Remove rows with missing values
df = df.dropna(subset=['subject', 'body', 'queue', 'type'])
df.head()

# Set your split ratios
TRAIN_RATIO = 0.9
VAL_RATIO = 0.09
TEST_RATIO = 0.01

PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared-data")
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)

# This list will hold all of our transformed data points.
transformed_data = []

def create_prompt(subject, body):
    """
    Creates a standardized prompt for the language model.
    """
    return f"A customer has submitted a support ticket. Please route it to the correct department.\n\nSubject: {subject}\n\nBody: {body}\n\nDepartment:"


# Iterate over each row of the DataFrame to create the prompt-completion pairs.
for index, row in df.iterrows():
    prompt = create_prompt(row['subject'], row['body'])
    # completion = row['type'] + ", " + row['queue']
    completion = row['queue']
    
    transformed_data.append({
        "input": prompt,
        "output": f"{completion}"
    })


random.shuffle(transformed_data)
n = len(transformed_data)

# Calculate split indices
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_data = transformed_data[:train_end]
val_data = transformed_data[train_end:val_end]
test_data = transformed_data[val_end:]

# Determine folder


def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

# Save each split
save_jsonl(train_data, os.path.join(PREPARED_DATA_DIR, "training.jsonl"))
save_jsonl(val_data, os.path.join(PREPARED_DATA_DIR, "validation.jsonl"))
save_jsonl(test_data, os.path.join(PREPARED_DATA_DIR, "test.jsonl"))

print(f"Total records: {n}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
print(f"Saved to {PREPARED_DATA_DIR}")

from pathlib import Path

import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed

# Define directories for intermediate artifacts
NEMO_MODELS_CACHE = "/home/aiscuser/yangwang/data/models-cache"
NEMO_DATASETS_CACHE = "/home/aiscuser/yangwang/data/data-cache"

os.environ["NEMO_DATASETS_CACHE"] = NEMO_DATASETS_CACHE
os.environ["NEMO_MODELS_CACHE"] = NEMO_MODELS_CACHE

# Configure the number of GPUs to use
NUM_GPU_DEVICES = 8

from getpass import getpass
from huggingface_hub import login

# login(token=getpass("Input your HF Access Token"))

import wandb

# WANDB_API_KEY = getpass("Your Wandb API Key:")

# wandb.login(key=WANDB_API_KEY)

# You can just as easily swap out the model with the 120B variant, or execute this on a remote cluster.

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=run.Config(llm.GPTOSSModel, llm.GPTOSSConfig20B),
        source="hf:///home/aiscuser/yangwang/data/gpt-oss-20b",
        overwrite=False,
    )
    
# Run your experiment locally
run.run(configure_checkpoint_conversion(), executor=run.LocalExecutor())

recipe = llm.gpt_oss_20b.finetune_recipe(
    name="gpt_oss_20b_finetuning",
    dir="/home/aiscuser/yangwang/nemo-experiments/",
    num_nodes=1,
    num_gpus_per_node=NUM_GPU_DEVICES,
    peft_scheme='none',  # 'lora', 'none' (for SFT)
    # peft_scheme='lora',  # 'lora', 'none' (for SFT)
)

from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

dataloader = run.Config(
        FineTuningDataModule,
        dataset_root=PREPARED_DATA_DIR,
        seq_length=2048,
        micro_batch_size=2,
        global_batch_size=16,
    )

# Configure the recipe
recipe.data = dataloader

# Visualize the dataloader
dataloader

from lightning.pytorch.loggers import WandbLogger

LOG_DIR = "/home/aiscuser/yangwang/nemo-experiments/results"
LOG_NAME = "nemo2_gpt_oss_sft_customer_ticket_routing"

def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=200,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    # Since WANDB was optional
    WANDB_API_KEY = None
    if WANDB_API_KEY is not None and WANDB_API_KEY != "":
        wandb_config = run.Config(
            WandbLogger, project="NeMo_LoRA_Customer_Ticket_Routing", name="Customer_Ticket_Routing"
        )
    else:
        wandb_config = None

    return run.Config(
        nl.NeMoLogger,
        name=LOG_NAME,
        log_dir=LOG_DIR,
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=wandb_config,
    )

recipe.log = logger()

logger()

def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(
            nl.RestoreConfig, path=f"nemo:///{NEMO_MODELS_CACHE}/gpt-oss-20b"
        ),
        resume_if_exists=True,
    )
    
recipe.resume = resume()

recipe.trainer.max_steps = 100
recipe.trainer.val_check_interval = 25
recipe.trainer.limit_val_batches = 2
recipe.optim.config.lr = 2e-4

# Let's visualize the recipe
recipe

run.run(recipe, executor=run.LocalExecutor(ntasks_per_node=NUM_GPU_DEVICES))

