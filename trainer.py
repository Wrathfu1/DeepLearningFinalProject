import os, sys, torch, numpy as np
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, set_seed
)

# ---------------------- sanity checks ----------------------
if len(sys.argv) < 2:
    print("usage: python npc_finetune.py <file.txt> [epochs]")
    sys.exit(1)

file_path = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

if not os.path.exists(file_path):
    print(f"file {file_path} not found")
    sys.exit(1)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

# reproducibility
SEED = 42
np.random.seed(SEED)
set_seed(SEED)

# ---------------------- load data --------------------------
dataset = load_dataset("text", data_files=file_path)["train"]
splits = dataset.train_test_split(test_size=0.1, seed=SEED)

# ---------------------- tokenizer / model ------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ---------------------- tokenisation -----------------------
def tok(batch):
    return tokenizer(batch["text"],
                     truncation=True,
                     padding="max_length",
                     max_length=128)

tok_splits = splits.map(tok, batched=True, remove_columns=["text"])
tok_splits.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ---------------------- training args ----------------------
batch = 1
accum = 4
out_dir = "./gpt2-npc-model"

args = TrainingArguments(
    output_dir=out_dir,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=accum,
    learning_rate=2e-5,
    optim="adamw_torch",
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    max_grad_norm=None,
    seed=SEED,
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
