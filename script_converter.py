import re
import sys
import unicodedata
from typing import List

import requests
from bs4 import BeautifulSoup


# Regular expressions to detect the "END CREDITS" block (all-caps text)
END_CRED_RE = re.compile(r'^[A-Z0-9 ,\-\’\'":]{50,}$')

# Regular expressions to detect character names and stage directions
NAME_RE = re.compile(r'^[A-Z][A-Z0-9\'\-\. ]{1,30}:?$')

#Regualar expressions to detect stage directions
STAGE_RE = re.compile(r'^(INT\.|EXT\.|OUTSIDE|CLOSE ON|FADE IN|FADE OUT|CUT TO|DISSOLVE TO)')

# Regular expressions to detect unnecessary characters
PAREN_RE = re.compile(r'^\(.*?\)$', flags=re.I)



#This function is used to normalize the text by removing unnecessary characters such as
#non-ascii characters and replacing them with their ascii equivalents
def normalize(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.replace("“", '"').replace("”", '"') \
               .replace("‘", "'").replace("’", "'") \
               .replace("—", "-").replace("\u00A0", " ")
    return text.strip()

#Function was another function to help remove unnecessary characters
#not similar to the normalize function but removed non-ascii characters 
#that do not have a specific ascii equivalent
#helps to flush out any corrupted lines
def isUncessary(line):
    if '�' in line:
        return True
    non_ascii = sum(ord(c) > 127 for c in line) #non_ascii characters are characters 
    #that are not in the ASCII range such as emojis
    return non_ascii / max(1, len(line)) > 0.15
#return true if the line contains more than 15% non-ascii characters. 
#I chose this number as a threshold to filter out lines that are likely to be corrupted or contain
#unnecessary characters.


#This function is used to fetch the script from the given URL
#It uses the requests library to send a GET request to the URL and returns the HTML content
def fetch_script(url):
    try:
        print("Connecting to the website...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        print("Successfully fetched the script")
        return response.text
    except requests.RequestException as e:
        print(f"Error: Unable to retrieve the movie script.") #Check if the URL is valid for reproducibility with other scripts
        return None


#Function to extract the dialogue lines from the HTML content
def extract_dialogue(html_content):
    print("Parsing the script content...")
    soup = BeautifulSoup(html_content, "html.parser")
    pre_tag = (soup.find("td", class_="scrtext") or soup).find("pre")

    if not pre_tag:
        return []

    # Clean up the text and split it into lines
    #raw text is the text that is extracted from the HTML
    #without any formatting or tags
    raw_lines = normalize(pre_tag.get_text()).splitlines()

    capture = False
    speaker = None
    buffer = []
    dialogue_lines = []

    # A function used to flush the text and only keep the dialogue lines
    # by flushing the buffer which is a list of lines that are not empty
    # and not unnecessary
    def flush():
        if speaker and buffer:
            dialogue_lines.append(f"{speaker.title()}: {' '.join(buffer).strip()}")
        buffer.clear()

    # Iterate through the lines and extract dialogue lines
    # after the "END CREDITS" section
    for line in raw_lines:
        line = line.strip() #strip() is used to remove leading and trailing whitespace
        if not line: 
            continue
        
        # Check if we are in the "END CREDITS" section
        if not capture:
            if END_CRED_RE.match(line):
                capture = True
            continue

        #Check to see if the line is empty or contains unnecessary characters (non-ascii characters)
        if isUncessary(line) or PAREN_RE.match(line) or STAGE_RE.match(line):
            flush()
            speaker = None
            continue

        # Check if the line is a character name or stage direction
        #Because stage directions look similar to character names, 
        # I used a regex to check if the line is a stage direction
        if NAME_RE.match(line.rstrip(':')):
            flush()
            speaker = line.rstrip(':').upper()
            continue

        #If we found that the line is a character name, we add it to the buffer
        if speaker:
            buffer.append(line) #buffer is a list of lines that are not empty and thus
            #contains all of our dialogue lines

    # Flush any remaining dialogue lines
    flush()
    if not dialogue_lines:
        print("No dialogue lines found after the 'END CREDITS' section.")
    else:
        print(f"Found {len(dialogue_lines)} lines of dialogue after 'END CREDITS'.")

    return dialogue_lines

#This function is used to save the dialogue lines to a file
def save_dialogue(dialogue_lines, output_file):
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for line in dialogue_lines:
                file.write(line + "\n")
        print(f"\nSuccess! Saved {len(dialogue_lines)} lines of dialogue to '{output_file}'.")
    except IOError as e:
        print(f"Error: Unable to save the file. Details: {e}")

#This function is the main function that is called when the script is run
#It handles user input and processes the script

def main():
    print("Welcome to the Movie Script Converter!")
    print("This tool extracts dialogue lines that appear after the 'END CREDITS' section of a movie script.")

    if len(sys.argv) < 2:
        print("\nUsage: python script_converter.py <IMSDB-URL> [output-file]\n")
        print("Example: python script_converter.py https://imsdb.com/scripts/Inception.html output.txt")
        return

    url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "train_data.txt"

    if not url.startswith("https://imsdb.com/scripts/"):
        print("Error: The URL must start with 'https://imsdb.com/scripts/'. Please provide a valid IMSDB script URL.")
        return

    print(f"\nProcessing the script from: {url}")
    html_content = fetch_script(url)
    if not html_content:
        return

    dialogue_lines = extract_dialogue(html_content)
    if not dialogue_lines:
        return

    save_dialogue(dialogue_lines, output_file)


if __name__ == "__main__":
    main()
