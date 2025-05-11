#This is the file to use the chatbot. Chatbot.py is 
#the main file to run the chatbot server

import requests
import sys


#URLs for the chatbot server 
#chat is for teh chatbot functionality
#eval is for the evaluation functionality
#The URLs are set to localhost and port 5000
URL_CHAT = "http://127.0.0.1:5000/chat"
URL_EVAL = "http://127.0.0.1:5000/evaluate"

# Ensure two arguments are provided: <Persona> and <Message>
if len(sys.argv) < 4:
    print("Usage: python communicator.py <TYPE> <Persona> <Message>")
    sys.exit(1)

style = sys.argv[1]
persona = sys.argv[2]
message = sys.argv[3]

#if the style is not eval or chat, we exit
if style == "eval":
    URL = URL_EVAL
else:
    URL = URL_CHAT


#code to communicate with the chatbot server
#This function sends a POST request to the chatbot server
def communicate_with_chatbot(persona, message):
    try:
        # Combine persona and message into a single input
        full_message = f"{persona}: {message}"
        response = requests.post(URL, json={"message": full_message}) # Send the request to the chatbot server
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the chatbot: {e}")
        return None

# This is the main function that runs when the script is executed
if __name__ == "__main__":
    response = communicate_with_chatbot(persona, message)
    if not response:
        print("Failed to get a response from the chatbot.")
        sys.exit(1)

    if style == "eval":
        print(f"Accuracy: {response['accuracy'] * 100:.2f}%\n")
        print("Classification report:")
        print(response["classification_report"], "\n")
        print("Confusion matrix:")
        for row in response["confusion_matrix"]:
            print(row)
    else:
        # Send the persona and message to the chatbot
        print(f"You : {message}")
        print(f"Chatbot response: {response['response']}")
        print(f"Persona: {response['persona']}")

