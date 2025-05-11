import os, sys, torch, numpy as np
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, set_seed
)
# ---------------------- file checks ------------------------

if len(sys.argv) < 2:
    print("usage: python npc_finetune.py <file.txt> [epochs]")
    sys.exit(1)

file_path = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3 # default to 3 epochs but will be user inputted

if not os.path.exists(file_path):
    print(f"file {file_path} not found")
    sys.exit(1)

print("CUDA available:", torch.cuda.is_available()) #detects whether CUDA is available for GPU acceleration
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

# reproducibility
SEED = 42
np.random.seed(SEED)
set_seed(SEED)

# ---------------------- load data --------------------------
dataset = load_dataset("text", data_files=file_path)["train"] #dataset is loaded from the file path
splits = dataset.train_test_split(test_size=0.1, seed=SEED) #splits the dataset into training and testing sets

# ---------------------- tokenizer / model ------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") #tokenizer is loaded from the pretrained gpt2 model
tokenizer.pad_token = tokenizer.eos_token #sets the pad token to the end of sentence token. Pad token is used to 
#pad the input sequences to the same length which allows for proper batch processing
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ---------------------- tokenisation -----------------------

#tokenisation is the process of converting the text into tokens
def tok(batch): 
    return tokenizer(batch["text"], #token_type_ids=None,
                     truncation=True, #set to True to truncate the text to the maximum length (truncate means to cut off the text)
                     padding="max_length", #pad the text to the maximum length (padding means to add zeros to the text matrix)
                     max_length=128)

#enables the splits of the dataset to be tokenized
tok_splits = splits.map(tok, batched=True, remove_columns=["text"])
tok_splits.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ---------------------- training args ----------------------
batch = 1
accum = 4
out_dir = "./gpt2-npc-model"

#Arguments for training the model
args = TrainingArguments(
    output_dir=out_dir, #output directory for the model
    overwrite_output_dir=True, #overwrite the output directory if it exists
    num_train_epochs=EPOCHS, #number of epochs to train for
    per_device_train_batch_size=batch, #batch size for training
    gradient_accumulation_steps=accum, #number of steps to accumulate gradients for
    learning_rate=2e-5, #learning rate for the optimizer
    optim="adamw_torch", #optimizer to use 
    save_steps=10_000, #number of steps to save the model
    save_total_limit=2, #maximum number of checkpoints to keep
    prediction_loss_only=True, #only save the prediction loss
    logging_steps=50, #number of steps to log the training progress
    eval_strategy="steps", #sets the evaluation strategy to steps
    eval_steps=500, #number of steps to evaluate the model
    max_grad_norm=None, #maximum gradient norm (None means no clipping)
    seed=SEED, #seed for random number generation
)

# ---------------------- collator ---------------------------
#Data collator is used to collate the data into batches
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#Trainer is used to train the model
#The trainer takes the model, training arguments, and the training and evaluation datasets
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_splits["train"],
    eval_dataset=tok_splits["test"],
    data_collator=collator,
)

# ---------------------- train & save -----------------------
trainer.train()
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("\nModel saved to", out_dir)
