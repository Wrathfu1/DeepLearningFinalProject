import sys
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset


#this is code to load the dataset given
# the file path
# and to process the data to be used for training
# and evaluation
#features label and text extraction 
def load_dataset(path):
    texts, labels = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): 
            continue
        speaker, _, utterance = line.partition(":") # code splits the speaker and utterance into three parts
        label = 0 if speaker.strip().lower().startswith("john") else 1 # 0 for John, 1 for Frank
        texts.append(utterance.strip())
        labels.append(label)
    return texts, labels

#Edge case detection for invalid command line arguments
if len(sys.argv) < 2:
    print("Usage: python SentimentAnalysis.py <input-file>")
    sys.exit(1)

input_file = sys.argv[1]
#text and label extraction for the dataset
#text is the utterance and label is the speaker
texts, labels = load_dataset(input_file)

if not texts or not labels:
    print("Error: The input file is empty or contains no valid data.")
    sys.exit(1)

#tokenizer and model loading
#BertTokenizer is used to tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #used to tokenize the text
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) #

#encodings is the tokenized text
#labels is the label for the text
encodings = tokenizer(texts, padding=True, truncation=True)
encodings["labels"] = labels
train_dataset = Dataset.from_dict(encodings)

# ---0------------------ training --------------------------

#TrainingArguments is used to set the training parameters
#The parameters include the output directory, evaluation strategy, learning rate, batch size, number of epochs, and weight decay
training_args = TrainingArguments(
    output_dir="./results", #output directory for the model
    eval_strategy="epoch", #evaluation strategy to use
    learning_rate=2e-5, #learning rate for the optimizer
    per_device_train_batch_size=16, #batch size for training
    per_device_eval_batch_size=16, #batch size for evaluation
    num_train_epochs=3, #number of epochs to train for
    weight_decay=0.01, #weight decay for the optimizer
)

#trainier is used to train the model
#The trainer takes the model, training arguments, and the training and evaluation datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
)

#Now we train the model
trainer.train()

# ---0------------------ save model --------------------------
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")