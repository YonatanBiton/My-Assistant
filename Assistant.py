import os
import subprocess
import webbrowser
import pyttsx3
import requests
from rapidfuzz import process
import spacy
import wikipediaapi
from datetime import datetime
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import sqlite3
import data_base_init
import soundfile as sf
import sounddevice as sd
import traceback
# Initialize spaCy NLP model
spacy_nlp = spacy.load("en_core_web_lg")
ruler = spacy_nlp.add_pipe("entity_ruler", before="ner")

# Initialize recognizer and text-to-speech engine
engine = pyttsx3.init()
# Set your API key (replace with your actual OpenAI API key)
weather_api_key = "d39f1c02345864641983ba1274530e37"

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
              (e.g., 'CITY', 'DATE') and the values.
    """
    try:
        doc = spacy_nlp(command)
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



def record_audio(duration=5, sample_rate=16000):
    """
    Record audio for a specified duration
    """
    print(f"Listening...")
    audio_data = sd.rec(int(sample_rate * duration),
                        samplerate=sample_rate,
                        channels=1,
                        dtype='float32')
    sd.wait()
    print("Recording finished!")
    return audio_data

def process_audio(audio_data):
    """
    Process audio data for Whisper
    """
    # Ensure audio is mono and float32
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize audio
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = np.clip(audio_data, -1.0, 1.0)

    return audio_data


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
        audio_array = process_audio(audio_array)
        input_features = whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        if transcription.strip():
            print(f"You said:{transcription}")
            return transcription
        return None
    except Exception as e:
        print("Error during speech recognition:")
        traceback.print_exc()
        return None
    
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

def find_closest_app(user_input, cursor):
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
        cursor.execute("SELECT app_name FROM whitelist_apps")
        app_names = [row[0] for row in cursor.fetchall()]
        print(f"app names: {app_names}")
        print(f"user_input: {user_input}")
        if app_names:
            best_match = process.extractOne(user_input.lower(), [app.lower() for app in app_names])
        if best_match and best_match[1] > 90: 
            cursor.execute("SELECT app_path FROM whitelist_apps WHERE LOWER(app_name) = ?", (best_match[0],))
            app_path_row = cursor.fetchone()
            if app_path_row:
                return app_path_row[0]  # Return the app path
            else:
                print("App path not found in the database for the matched app.")
        
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


def close_app(app_name):
    """
    Attempts to close an application by killing its process using the provided `app_name`.
    Args:
        app_name (str): The name of the application to close.
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
        str: A short summary of the Wikipedia page, consisting of the first two sentences.
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
    

def check_general_question_in_command(command, cursor):
    """
    Checks if the given command contains a general question and extracts the query.
    Args:
        command (str): The command or input string that may contain a general question.
    Returns:
        str: The extracted query from the command if a matching question phrase is found.
    """
    # Query to get all phrases from the question_starters table
    cursor.execute("SELECT phrase FROM question_starters")
    phrases = cursor.fetchall()
    for phrase_tuple in phrases:
        phrase = phrase_tuple[0]
        if phrase.lower() in command.lower():  # Case insensitive matching
            return True

    return None  # Return None if no phrase match is found



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
        question = question[1:]
        doc = spacy_nlp(question.upper())
        print("Entities:", doc.ents)
        subject = None
        for ent in doc.ents:
            # Check for specific named entities (e.g., PRODUCT, FAC, LOC, etc.)
            if ent.label_ in ['PRODUCT', 'FAC', 'DATE', 'LOC', 'GPE', 'PERSON', 'ORG', 'EVENT', 'APP']:
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
    '''
    Matches a website name in the user's command and retrieves its URL from the database.
    Args:
        command (str): The user's voice or text command containing a potential website name.
        cursor (sqlite3.Cursor): A database cursor to execute SQL queries.

    Returns:
        str: The URL of the matched website if found, otherwise None.
    '''
        # Query to get all site names and URLs from the database
    cursor.execute("SELECT website_name, url FROM common_websites")
    websites = cursor.fetchall()
    for site_name, site_url in websites:
        if site_name.lower() in command.lower():  # Case insensitive matching
            print(site_name)
            return site_name,site_url
    return None, None
    

def insert_new_app(app_name, app_path, cursor, db_conn):
    '''
    Inserts a new application into the 'whitelist_apps' table in the database.
    Args:
        app_name (str): The name of the application to be added.
        app_path (str): The file path of the application to be added.
        cursor (sqlite3.Cursor): A database cursor to execute SQL queries.
        db_conn (sqlite3.Connection): A database connection to commit changes.
    '''
    try:
    # Check if the app already exists
        cursor.execute('''
        SELECT COUNT(*) FROM whitelist_apps WHERE app_name = ?
        ''', (app_name,))
        result = cursor.fetchone()
        if result[0] > 0:
            speak(f"{app_name} is already exists to you want to over right it? Yes/No")
            user_confirmation = listen()
            print(user_confirmation)
            if user_confirmation and ("no" in user_confirmation or "No" in user_confirmation or "nope" in user_confirmation):
                speak(f"{app_name} wasn't added to your app list")
                return 
        else:
            # Insert the new app if it doesn't exist
            cursor.execute('''
            INSERT INTO whitelist_apps (app_name, app_path)
            VALUES (?, ?)
            ''', (app_name, app_path))
            speak(f"Added {app_name} to your app list.")
            db_conn.commit()  # Commit only if new app is added

    except sqlite3.Error as e:
        print(f"Error checking or inserting whitelist apps: {e}")

def add_ruler_to_nlp(cursor):
    '''
    Dynamically adds application-related patterns to an NLP entity ruler for recognition.
    Args:
        cursor (sqlite3.Cursor): A database cursor used to fetch application names from the `whitelist_apps` table.
    '''
    # Fetch app names
    cursor.execute("SELECT app_name FROM whitelist_apps")
    apps = [row[0] for row in cursor.fetchall()]
    # Define patterns dynamically
    patterns = [{"label": "APP", "pattern": app} for app in apps]
    ruler.add_patterns(patterns)

def execute_command(command, cursor, db_conn):
    """
    Processes and executes the given user command.
    Args:
        command (str): The user command to execute.
        cursor: Database cursor for querying app and website data.
        db_conn: Database connection for app management.
    Returns:
        None: Interacts via voice or browser without returning values.
    """
    flag = 0
    entities = extract_entities(command.lower())  # Extract entities using spaCy
    print(entities)
    if 'weather' in command:  # If the command contains 'weather', it fetches the weather
        city = entities.get("GPE")  # Get city (Geopolitical Entity) from entities
        if city:
            weather_info = get_weather(city)
            speak(weather_info)
        else:
            speak("Didn't recognize a city, please try again.")
            flag = 1

    #wikipidia requests
    elif check_general_question_in_command(command, cursor):
        subject = extract_subject_from_question(command)
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
    elif 'open'  in command.lower():  
        #extraction of the app name from the command
        app_name = entities.get("APP")
        website_name , website_url = website_opener(command, cursor)
        #handling common website openning
        if website_url:
            webbrowser.open(website_url)
            speak(f"Opening {website_name}.")
        elif website_name != None:
            speak(f"Sorry, I couldn't find {website_name} in my website table.")
        #handling computer app openning
        if app_name:
            app_path = find_closest_app(app_name, cursor)
            open_app(app_path)
            speak(f"Opening {app_name}.")
        elif app_name != None:
            speak(f"Sorry, I couldn't find {app_name} in your apps.")
        elif website_name == None and app_name == None:
            speak("Couldnt understand the application or website you were looking for. please try again")

    #handling app closing
    elif 'close' in command:
        app_name = entities.get("APP")
        print(app_name)
        app_path = find_closest_app(app_name,cursor)
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

        insert_new_app(app_name, app_path, cursor, db_conn)
    
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
    Initializes and runs the virtual assistant.
    - Sets up the database and loads the Whisper speech recognition model.
    - Configures NLP patterns for app-specific commands.
    - Greets the user based on the time of day.
    - Continuously listens for and executes commands until "exit" or "bye" is detected.
    - Handles graceful shutdown on user exit or interruption.

    Exceptions:
        - Logs errors during initialization or runtime interruptions.
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

    #adding ruler to nlp
    add_ruler_to_nlp(cursor)

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
    
    try:
        while True:
            command = listen()
            if command:
                if "exit" in command or "bye" in command:
                    speak("Goodbye!")
                    db_conn.close()
                    break
                execute_command(command, cursor, db_conn)
    except KeyboardInterrupt:
        print("\nStopping assistant...")

if __name__ == "__main__":
    assistant_main()
