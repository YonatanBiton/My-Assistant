import os
import json
import subprocess
import webbrowser
import pyttsx3
import requests
from rapidfuzz import process
import spacy
import wikipediaapi
from datetime import datetime
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import pyaudio
import re
import sqlite3
import data_base_init
# Initialize spaCy NLP model
nlp = spacy.load("en_core_web_lg")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
# Load the RoBERTa model fine-tuned for NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer)




# Initialize recognizer and text-to-speech engine
engine = pyttsx3.init()

# Set your API key (replace with your actual OpenAI API key)
weather_api_key = "d39f1c02345864641983ba1274530e37"
CACHE_FILE = "user_apps.json"


# Dictionary of general question starters
question_starters = {
    "who": ["who is", "who was", "who are", "who were"],
    "what": ["what is", "what was", "what are", "what were"],
    "where": ["where is", "where was", "where are", "where were"],
    "when": ["when is", "when was", "when are", "when were"],
    "why": ["why is", "why was", "why are", "why were"],
    "how": ["how is", "how was", "how are", "how were"]
}
# list of apps found in dir that has a lot of .exe files that arent important
WHITELIST_APPS = {
    "notepad": r"C:\Windows\System32\notepad.exe",
    "calculator": r"C:\Windows\System32\calc.exe",
    "snippingtool": r"C:\Windows\System32\SnippingTool.exe",
    "paint": r"C:\Windows\System32\mspaint.exe",
}
#comon website url's
common_websites = {
    "search_engines": {
        "google": "https://www.google.com",
        "bing": "https://www.bing.com",
        "duckduckgo": "https://www.duckduckgo.com"
    },
    "social_media": {
        "facebook": "https://www.facebook.com",
        "twitter": "https://www.twitter.com",
        "instagram": "https://www.instagram.com",
        "linkedin": "https://www.linkedin.com",
        "reddit": "https://www.reddit.com"
    },
    "entertainment": {
        "youtube": "https://www.youtube.com",
        "netflix": "https://www.netflix.com",
        "hulu": "https://www.hulu.com",
        "spotify": "https://www.spotify.com",
        "twitch": "https://www.twitch.tv"
    },
    "news": {
        "bbc": "https://www.bbc.com",
        "cnn": "https://www.cnn.com",
        "nytimes": "https://www.nytimes.com",
        "the_guardian": "https://www.theguardian.com",
        "reuters": "https://www.reuters.com"
    },
    "shopping": {
        "amazon": "https://www.amazon.com",
        "ebay": "https://www.ebay.com",
        "walmart": "https://www.walmart.com",
        "etsy": "https://www.etsy.com",
        "aliexpress": "https://www.aliexpress.com"
    },
    "education": {
        "wikipedia": "https://www.wikipedia.org",
        "khan_academy": "https://www.khanacademy.org",
        "coursera": "https://www.coursera.org",
        "edx": "https://www.edx.org",
        "udemy": "https://www.udemy.com"
    },
    "productivity": {
        "google_drive": "https://drive.google.com",
        "dropbox": "https://www.dropbox.com",
        "notion": "https://www.notion.so",
        "trello": "https://www.trello.com",
        "slack": "https://www.slack.com"
    },
    "developer_tools": {
        "github": "https://www.github.com",
        "stackoverflow": "https://stackoverflow.com",
        "gitlab": "https://www.gitlab.com",
        "docker_hub": "https://hub.docker.com",
        "npm": "https://www.npmjs.com"
    }
}
SIZE_LIMIT = 100 * 1024  # 100 KB size limit for valid executables

#using spacy NLP to extract the entitie from the command
#for exmaple: what is the weather in paris {city : paris}
def extract_entities(command):
    """
    Extract entities from the user command using spaCy's NLP pipeline.
    Args:
        command (str): The user command in the form of a string.
    Returns:
        dict: A dictionary where the keys are the entity labels 
              (e.g., 'CITY', 'DATE') and the values are the 
              corresponding entity values (e.g., 'Paris').
    Example:
        command = "What is the weather in Paris?"
        extract_entities(command)
        # Output: {'CITY': 'Paris'}
    """
    try:
        doc = nlp(command)
        entities = {ent.label_: ent.text for ent in doc.ents}
        return entities
    except Exception as e:
        print(f"An error occurred while extracting entities: {e}")
        return {}



def initialize_whisper(model_name="openai/whisper-small.en"):
    """
    Initialize the Whisper model for speech recognition.
    Args:
        model_name (str): The name of the Whisper model to load. 
                          Default is "openai/whisper-small.en".
    Returns:
        tuple: A tuple containing:
            - model: The loaded Whisper model.
            - processor: The processor for preprocessing audio input.
            - device: The device ("cuda" or "cpu") used for model inference.
    """
    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    return model, processor, device


def record_audio(duration=5, sample_rate=16000, threshold=0.01):
    """
    Record audio from the microphone and detect speech using the is_speaking function.
    Args:
        duration (int): The duration (in seconds) to record audio.
        sample_rate (int): The sample rate for audio recording (default is 16000 Hz).
        threshold (float): The threshold value used to detect speech. 
    Returns:
        numpy.ndarray or None: 
            - A numpy array containing the concatenated speech frames if speech is detected.
            - `None` if no speech is detected during the recording.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    print("Listening...")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16) / 32768.0  # Normalize to [-1, 1]
        
        # Check if the audio contains speech
        if is_speaking(audio_data, threshold):
            print("User is speaking...")
            frames.append(audio_data)
    print("Processing...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    if frames:
       
        return np.concatenate(frames)
    else:
        print("No speech detected.")
        return None  # No speech detected

def listen():
    """
    Listens to the user's speech, processes it using the Whisper model, and returns the transcribed text.
    Returns:
        str or None:
            - A lowercase string of the transcribed speech if speech is successfully detected.
            - `None` if an error occurs or no speech is detected.
    """
    global whisper_model, whisper_processor, device
    try:
        audio_array = record_audio()
        input_features = whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        if transcription.strip():
            print(f"You said: {transcription}")
            return transcription.lower()
        return None
    except Exception as e:
        import traceback
        print("Error during speech recognition:")
        traceback.print_exc()
        return None


def is_speaking(audio_data, threshold=0.01):
    """
    Determine if the user is speaking based on audio amplitude.
    :param audio_data: Numpy array of audio samples
    :param threshold: Amplitude threshold to detect speech
    :return: Boolean, True if speaking, False otherwise
    """
    rms = np.sqrt(np.mean(audio_data**2))  # Calculate Root Mean Square (RMS)
    return rms > threshold


def speak(text):
    """
    Speaks the given text using the text-to-speech engine.
    Args:
        text (str): The text to be spoken.
    Example:
        speak("Hello, how can I assist you?")
    """
    engine.say(text)
    engine.runAndWait()


def load_cached_paths():
    """
    Loads cached paths from a file if it exists, or creates the file if it doesn't.
    Returns:
        dict: The cached paths as a dictionary. If the file didn't exist, returns an empty dictionary.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    else:
        # Create the cache file with an empty dictionary
        with open(CACHE_FILE, "w") as file:
            json.dump({}, file)
        speak("Cache file not found. A new cache file has been created.")
    return {}


def save_cached_paths(app_paths):
    """
    Saves the given application paths to a cache file in JSON format.
    Args:
        app_paths (dict): A dictionary containing the application paths to be saved. 
    """
    with open(CACHE_FILE, "w") as file:
        json.dump(app_paths, file)


def find_closest_app(user_input, app_dict, cursor):
    """
    Finds the closest matching application name from the provided dictionary using fuzzy string matching.
    Args:
        user_input (str): The input string from the user, which will be matched against the application names.
        app_dict (dict): A dictionary where keys are application names (strings) and values are application paths or other related information.
    Returns:
        str or None: The value associated with the closest matching key in `app_dict`, or `None` if no suitable match is found.
    Example:
        user_input = "googl"
        app_dict = {"google": "/path/to/google", "gmail": "/path/to/gmail"}
        closest_app = find_closest_app(user_input, app_dict)
        print(closest_app)  # Output: "/path/to/google"
    """
    try:
        # Ensure user_input is a string and perform fuzzy matching
        #best_match = process.extractOne(user_input.lower(), app_dict.keys())
        cursor.execute("SELECT app_name FROM apps")
        app_names = [row[0] for row in cursor.fetchall()]
        best_match = process.extractOne(user_input.lower(), [app.lower() for app in app_names])
        if best_match and best_match[1] > 90:  # Adjust threshold
            return app_dict[best_match[0]]
        
    except Exception as e:
        # Catch all exceptions to prevent the program from crashing
        print(f"An error occurred: {e}")
        return None

def open_app(app_path):
    """
    Tries to open an application located at the specified `app_path` and handles any errors that may occur.
    Args:
        app_path (str): The file path to the application to be opened. This should be a valid path to an executable file.
    Returns:
        subprocess.Popen or None: The `Popen` result if the application is opened successfully, otherwise `None`.
    """
    try:
        print(f"Attempting to open: {app_path}")
        app_path = f'"{app_path}"'
        result = subprocess.Popen(app_path,  creationflags=subprocess.DETACHED_PROCESS)
        print(f"App opened successfully: {app_path}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Failed to open {app_path}: {e}")
    except Exception as e:
        print(f"Error opening app: {e}")

def find_app_in_whitelist(app_name):
    """
    Searches for an app in the whitelist using fuzzy matching on the provided `app_name`.
    Args:
        app_name (str): The name of the app to search for in the whitelist. This string will be compared against the keys in the whitelist.
    Returns:
        str or None: The path to the app in the whitelist if a match is found, otherwise `None`.
    """
    # Use fuzzy matching to handle variations in user input
    best_match = process.extractOne(app_name.lower(), WHITELIST_APPS.keys())
    if best_match and best_match[1] > 90:  # Adjust threshold as needed
        return WHITELIST_APPS[best_match[0]]
    return None


def close_app(app_name):
    """
    Attempts to close an application by killing its process using the provided `app_name`.
    Args:
        app_name (str): The name of the application to close. This should be the name of the executable (without the `.exe` extension).
    """
    try:
        subprocess.run(f"taskkill /f /im {app_name}.exe", shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to close {app_name}. Process not found.")
    except Exception as e:
        print(f"Error closing {app_name}: {e}")


def get_weather(city):
    """
    Fetches and returns the current weather information for a given city using the OpenWeatherMap API.
    Args:
        city (str): The name of the city for which to fetch the weather information.

    Returns:
        str: A string containing the current weather information (e.g., "clear sky", "light rain").
             If the weather information cannot be fetched, it returns a fallback error message.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        location = data.get('name', 'Unknown city')
        temperature = data.get('main', {}).get('temp', 'No temperature data')
        weather_description = data.get('weather', [{}])[0].get('description', 'No description available')
        temperature = str(int(float(temperature) - 273.15))
        return f"The weather in {location}, is {temperature}°C with {weather_description}."
    else:
        return "Sorry, I couldn't fetch the weather information."

def get_wikipedia_summary(query):
    """
    Retrieves a brief summary from Wikipedia for a given query.
    Args:
        query (str): The topic or keyword for which to retrieve the Wikipedia summary.

    Returns:
        str: A short summary of the Wikipedia page, consisting of the first two sentences, or an error message 
             if the page doesn't exist or an error occurs.
    """
    user_agent = "MyAlexaBot/1.0 (yuyu82114@gmail.com)"
    wiki = wikipediaapi.Wikipedia(user_agent, 'en')
    try:
        # Search for the page
        page = wiki.page(query)
        
        # If the page exists, get the first two sentences
        if page.exists():
           
            summary = page.summary.split('.')[0:2]  # Get first two sentences
            return '. '.join(summary) + '.'  # Join and return the summary
        else:
            return "Sorry, I couldn't find a Wikipedia page for that topic."
    except Exception as e:
        return f"An error occurred: {e}"
    

def check_general_question_in_command(command):
    """
    Checks if the given command contains a general question and extracts the query.
    Args:
        command (str): The command or input string that may contain a general question.

    Returns:
        str: The extracted query from the command if a matching question phrase is found,
             or None if no match is found.
    """
    for key, phrases in question_starters.items():
        for phrase in phrases:
            if phrase in command.lower():
                query = command[len(phrase):].strip()
                return query
    return None


def extract_subject_from_question(question):
    """
    Extracts the subject of a question using spaCy's named entity recognition and noun phrase extraction.
    Args:
        question (str): The question string from which to extract the subject.
    Returns:
        str: The extracted subject, which could be a named entity or the first noun phrase found, or 
             None if no suitable subject is identified.
    """
    try:
        # Process the question with spaCy
        doc = nlp(question.upper())
        print("Entities:", doc.ents)
        subject = None
        for ent in doc.ents:
            # Check for specific named entities (e.g., PRODUCT, FAC, LOC, etc.)
            if ent.label_ in ['PRODUCT', 'FAC', 'DATE', 'LOC', 'GPE', 'PERSON', 'ORG', 'EVENT']:
                subject = ent.text
                break

        # If no named entity is found, fallback to noun phrases
        if not subject:
            for np in doc.noun_chunks:
                subject = np.text
                break

        return subject

    except Exception as e:
        # Handle exceptions and print an error message
        print(f"An error occurred while processing the question: {e}")
        return None


def website_opener(command, cursor):
    """
    This function iterates through a dictionary of common websites and returns the URL of the site
    that matches the name found in the user's command.
    Args:
        command (str): The user's command, e.g., "Open Facebook" or "Launch Spotify".
    Returns:
        str: The URL of the website that matches the site name in the command.
             If no match is found, the function implicitly returns None.
    """
        # Query to get all site names and URLs from the database
    cursor.execute("SELECT website_name, url FROM common_websites")
    websites = cursor.fetchall()
    for site_name, site_url in websites:
        if site_name.lower() in command.lower():  # Case insensitive matching
            return site_url
    return None
    

def extract_app_name(command):
    """
    Extract the app name from the user command by removing action words and 
    identifying the application name using named entity recognition.
    Args:
        command (str): The user's command, e.g., "Open Facebook" or "Launch Spotify".

    Returns:
        str: The extracted app name, either from the named entities or the remaining command.
    """
    action_words = ['open', 'close', 'launch', 'start', 'run']
    for word in action_words:
        command = command.replace(word, "")  
    # Extract entities (ORG or other) from the command
    entities = extract_entities(command)
    app_name = entities.get("ORG", command.strip())  # Default to the stripped command if no entity is found
    
    return app_name


def extract_entities_with_roberta(text):
    """
    Extract named entities from the input text using a RoBERTa model fine-tuned for Named Entity Recognition (NER).
    Args:
        text (str): The input text from which named entities need to be extracted.
    Returns:
        list: A list of dictionaries representing the detected entities.

    Example:
        text = "Elon Musk founded SpaceX in California."
        entities = extract_entities_with_roberta(text)
        # Output: 
        # Entity: Elon Musk, Label: I-PER
        # Entity: SpaceX, Label: I-ORG
        # Entity: California, Label: I-LOC
    """
    # Normalize the text to title case for better entity recognition
    text = text.title()

    entities = nlp(text)
    for entity in entities:
        print(f"Entity: {entity['word']}, Label: {entity['entity']}")
    
    return entities

def extract_entity_string(entities):
    """
    Extract a string of organization names from a list of entities.
    Args:
        entities (list): A list of dictionaries representing the detected entities (the entity label, e.g., 'ORG', 'PER', 'LOC').
    Returns:
        str: A string of organization names concatenated together, separated by spaces.

    Example:
        entities = [
            {'word': 'Elon Musk', 'entity': 'PER'},
            {'word': 'SpaceX', 'entity': 'ORG'},
            {'word': 'California', 'entity': 'LOC'}
        ]
        org_string = extract_entity_string(entities)
        # Output: "SpaceX"
    """
    org_entities = [entity['word'].replace("▁", "") for entity in entities if 'ORG' in entity['entity']]
    return " ".join(org_entities)


def extract_word_after_command(command, target_word):
    """
    Extract the word that appears immediately after the target command in the sentence.
    Args:
        command (str): The input sentence or command from which the word after the target command will be extracted.
        target_word (str): The target word (e.g., 'open', 'close') that the function will search for in the command.
    Returns:
        str or None: The word immediately after the target word in the command, or None if no word is found.

    Example:
        command = "Please open Chrome"
        target_word = "open"
        result = extract_word_after_command(command, target_word)
        # Output: "Chrome"
    """
    match = re.search(r'\b' + re.escape(target_word) + r'\b (\w+)', command)
    if match:
        return match.group(1)
    else:
        return None


def insert_new_app(app_name, app_path, cursor, db_conn):
    try:
        cursor.execute('''
            INSERT INTO whitelist_apps (app_name, app_path)
            VALUES (?, ?)
            ''', (app_name, app_path))
        print("added new app to applist")
    except sqlite3.Error as e:
        print(f"Error inserting whitelist apps: {e}")

    db_conn.commit()



def execute_command(command, installed_apps, cursor, db_conn):
    """
    This function manages and executes different commands received by the assistant. It processes the command, extracts relevant 
    information (e.g., entities, app names), and performs actions.

    Args:
        command (str): The user command that the assistant will execute.
        installed_apps (dict): A dictionary of installed apps and their paths used to find and execute apps.
    Returns:
        None: The function interacts with the user via voice or browser but doesn't return anything.
    """
    flag = 0
    entities = extract_entities(command)  # Extract entities using spaCy
    if 'weather' in command:  # If the command contains 'weather', it fetches the weather
        city = entities.get("GPE")  # Get city (Geopolitical Entity) from entities
        if city:
            weather_info = get_weather(city)
            speak(weather_info)
        else:
            speak("Didn't recognize a city, please try again.")
            flag = 1

    #wikipidia requests
    elif check_general_question_in_command(command):
        print(command)
        subject = extract_subject_from_question(command)
        print(subject)
        response = get_wikipedia_summary(subject)
        if "couldn't find" in response.lower():  # If Wikipedia doesn't return results
            speak(f"I couldn't find an answer on Wikipedia. Do you want me to search for it on Google?")
            user_confirmation = listen()
            if user_confirmation and ("yes" in user_confirmation or "sure" in user_confirmation or "yeah" in user_confirmation):
                webbrowser.open(f"https://www.google.com/search?q={command}")
                speak("Opening Google with your query.")
            else:
                speak("Alright, let me know if you need help with anything else.")
        else:
            speak(response)
            flag = 1

    #handling app opnening
    elif 'open' in command:  
        #extraction of the app name from the command
        app_name = extract_entities_with_roberta(command)
        app_name = extract_entity_string(app_name)
        if app_name is not None:
            app_name = extract_word_after_command(command,'open')
        #trying to find the app name in side the installed apps dic
        app_path = find_closest_app(app_name, installed_apps, cursor)
        whitelist_app_path = find_app_in_whitelist(app_name)
        website_url = website_opener(command, cursor)
        print("app name: "+app_name)
        #handling common website openning
        if website_url:
            webbrowser.open(website_url)
            speak(f"Opening {app_name}.")
        #handling computer app openning
        elif app_path:
            open_app(app_path)
            speak(f"Opening {app_name}.")
        else:
            speak(f"Sorry, I couldn't find {app_name} in your apps.")
            flag = 1

    #handling app closing
    elif 'close' in command:
        app_name = extract_entities_with_roberta(command)
        app_name = extract_entity_string(app_name)
        if app_name is not None:
            app_name = extract_word_after_command(command,'close')
        print(app_name)
        app_path = find_closest_app(app_name, installed_apps)
        if(app_path):
            close_app(app_name)
            speak(f"Closing app {app_name}.")
        else:
            speak(f"Sorry, I couldn't find an app named {app_name}.")

    #handling add app to app list
    elif 'add' in command:
        """Allow user to add an application to the whitelist."""
        app_name = input("Enter the application name: ").strip().lower()
        app_path = input("Enter the full path to the application: ").strip()
        # Validate inputs
        if not app_name or not app_path:
            print("Invalid input. Both name and path are required.")
            return
        # Check if app name already exists
        if app_name in installed_apps:
            overwrite = input(f"{app_name} already exists. Overwrite? (yes/no): ").strip().lower()
            if overwrite != "yes":
                print("Operation canceled.")
        installed_apps[app_name] = app_path
        insert_new_app(app_name, app_path, cursor, db_conn)
        save_cached_paths(installed_apps)
        speak(f"Added {app_name} to the whitelist.")
    
    #if the assistant didnt understand the command flag = 1 to ask the user if he wants to look up the command on google
    else:
        speak("Sorry didn't understand the command.")
        flag = 1
    #in case assistant didn't understand the command, ask user to look up the command on google
    if flag == 1:
        speak(f"Do you want me to look up{command} in google?")
        user_confirmation = listen()
        print(user_confirmation)
        if user_confirmation and ("yes" in user_confirmation or "sure" in user_confirmation or "yeah" in user_confirmation):    
            webbrowser.open(f"https://www.google.com/search?q={command}")
            speak("Opening Google with your query.")
        else:
            speak("Not looking it up in Google, let me know if you need help with anything else.")

def get_time_of_day():
    """
    Determines the current time of day based on the system's hour and returns an appropriate greeting.
    Returns:
        str: A string representing the time of day based on the current hour.
    Time of Day Mapping:
        - "Morning": From 5:00 AM to 11:59 AM.
        - "Afternoon": From 12:00 PM to 5:59 PM.
        - "Evening": From 6:00 PM to 8:59 PM.
        - "Night": From 9:00 PM to 4:59 AM.
    """
    current_hour = datetime.now().hour

    if 5 <= current_hour < 12:
        return "Morning"
    elif 12 <= current_hour < 18:
        return "Afternoon"
    elif 18 <= current_hour < 21:
        return "Evening"
    else:
        return "Night"


def assistant_main():
    """
    Main loop for the assistant that handles initialization, listens for commands, and executes actions.
    The function:
        - Initializes Whisper for speech recognition.
        - Loads the cached list of installed apps.
        - Greets the user based on the time of day.
        - Continuously listens for commands and executes them.
        - Allows the assistant to exit gracefully when "exit" or "bye" is said.
    Returns:
        None: The function operates continuously until the user terminates the program.
    """
    global whisper_model, whisper_processor, device
    

    # Initialize DataBase
    try:
        if not os.path.isfile("assistant_data.db"):
            data_base_init.main()
        db_conn = data_base_init.create_connection("assistant_data.db")
        cursor = db_conn.cursor()
    except Exception as e:
        print(f"Error initializing data base: {e}")


    # Initialize Whisper
    try:
        whisper_model, whisper_processor, device = initialize_whisper()
    except Exception as e:
        print(f"Error initializing Whisper: {e}")
        return

    # Load cached apps
    cached_apps = load_cached_paths()

    if not cached_apps:
        speak("could not load app file")
    else:
        installed_apps = cached_apps

    # Main loop to listen for commands
    start_time = get_time_of_day()
    if(start_time == "Morning"):
        speak("Good Morning ")
    elif start_time == "Afternoon":
        speak("Good after noon ")
    elif start_time == "Evening":
        speak("good evening ")
    elif start_time == "Night":
        speak("good night ")
    print(extract_entities_with_roberta("can you open fl64  ?"))
    try:
        while True:
            command = listen()
            if command:
                if "exit" in command or "bye" in command:
                    speak("Goodbye!")
                    db_conn.close()
                    break
                execute_command(command, installed_apps, cursor, db_conn)
    except KeyboardInterrupt:
        print("\nStopping assistant...")

if __name__ == "__main__":
    assistant_main()
