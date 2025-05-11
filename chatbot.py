
from flask import Flask, request, jsonify
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from datasets import load_dataset
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_path = "gpt2-npc-model"     # Path to the trained model
encoder_path = "./sentiment_model"  # Path to the sentiment analysis model

from transformers import BertTokenizer, BertForSequenceClassification   
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sentiment_tokenizer = BertTokenizer.from_pretrained(encoder_path)
sentiment_model     = BertForSequenceClassification.from_pretrained(encoder_path).to(device)

tokenizer = GPT2Tokenizer.from_pretrained(decoder_path)
model     = GPT2LMHeadModel.from_pretrained(decoder_path).to(device)

#sets a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


@app.route('/chat', methods=['POST'])
def chat():
    #user input is the message sent by the user
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # ----- encoder processing + persona override -----

    #if the user input is in the format "Frank: <message>" or "John: <message>"
    # we override the personality prefix
    if ':' in user_input and user_input.split(':', 1)[0] in ("Frank", "John"):
        personality_prefix, user_input = user_input.split(':', 1) #the user input is split into two parts based 
        #on the first occurrence of ":" and the first part is assigned to personality_prefix
        personality_prefix = personality_prefix.strip()
        user_input         = user_input.strip()
    else:
        
        #torch.no_grad() is used to disable gradient calculation
        # which reduces memory consumption and thus boosts
        #computational efficiency
        with torch.no_grad():

            # Tokenize the user input and get the model's predictions
            enc = sentiment_tokenizer(user_input, return_tensors='pt', padding=True).to(device)
            # The model's output is the logits which are the raw predictions
            logits = sentiment_model(**enc).logits
            # The argmax of the logits gives us the predicted class and we convert it to a Python integer for
            # the identification of the personality
            bert_id = torch.argmax(logits, dim=-1).item()
            #personality prefix either Frank or John (Main characters of their respective movies)
            personality_prefix = "Frank" if bert_id == 0 else "John"
        print(f"Personality: {personality_prefix}")


    # The sentiment analysis model is used to determine the personality of the user
    sentiment_output = f"[Sentiment={personality_prefix}]\nUser: {user_input}\nBot:"
    # The sentiment output is a formatted string that includes the personality prefix and the user input
    inputs = tokenizer(
        sentiment_output,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # ----- decoder processing -----

    
    with torch.no_grad():

        # Generate a response using the GPT-2 model
        gpt_out = model.generate(
            **inputs, #is the input to the model
            max_length=inputs["input_ids"].shape[-1] + 50, #the max length of the output given by the user's input 
            no_repeat_ngram_size=2, # prevents the model from repeating n-grams of size 2. Chose 2 as a default
            num_return_sequences=1, # number of sequences to return
            do_sample=True, # enables sampling which allows the model to generate more diverse outputs
            top_k=50, # limits the sampling pool to the top 50 tokens
            top_p=0.95, #considers the cumulative probability of the top tokens
            temperature=0.7, #helps control the randomness of the output
            pad_token_id=tokenizer.eos_token_id #end of sequence token id
        )

        # Decode the generated token IDs to text
        gen_ids  = gpt_out[0, inputs["input_ids"].shape[-1]:].tolist()
        # The generated token IDs are sliced to exclude the input part allowing us to keep only the generated response
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"Response: {response}")

    return jsonify({"response": response, "sentiment": personality_prefix})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    # load the first 100 examples from SST-2's validation split

    #Using a pre-trained model from the Hugging Face library
    # and the SST-2 dataset from the GLUE benchmark
    #will help provide a more accurate and efficient evaluation
    #of the sentiment analysis model.
    # This is a sample dataset for sentiment analysis
    ds = load_dataset("glue", "sst2", split="validation[:100]")
    texts = ds["sentence"]

    #helps us assign the labels to the dataset
    # The labels are 0 for negative and 1 for positive sentiment (based on speaker)
    labels = ds["label"]  # 0=negative, 1=positive

    # tokenize & run the BERT model

    #converrts the text into a format that the model can understand
    enc = sentiment_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)


    #logits are the output of the model
    logits = sentiment_model(**enc).logits

    #forms a prediction by taking the argmax of the logits. Taking argmax conceptually means to
    #choose the index of the maximum value in the logits tensor
    preds = torch.argmax(logits, dim=-1).cpu().numpy()

    #accuracy score
    acc = accuracy_score(labels, preds)
    #confusion matrix for evaluating the performance of the model
    cm  = confusion_matrix(labels, preds)

    #classification report gives us a summary of the performance of the model
    rpt = classification_report(labels, preds, target_names=["Negative","Positive"])

    return jsonify({
      "accuracy": acc,
      "confusion_matrix": cm.tolist(),
      "classification_report": rpt
    })

if __name__ == '__main__':
    port = 5000
    print(f"Chatbot is running on port {port}")
    app.run(host='127.0.0.1', port=port, debug=True)