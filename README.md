# Voice Assistant with NLP and Command Control
This project is a voice-controlled assistant built with Python that integrates multiple functionalities, including speech recognition, natural language processing (NLP), weather information, Wikipedia queries, application launching, and more. The assistant uses various NLP models such as spaCy, Whisper, and RoBERTa to process and understand user commands.

##Features
- Speech Recognition: Convert spoken commands into text using Whisper and Google Speech-to-Text API.
- Natural Language Processing (NLP): Understand user queries using spaCy and RoBERTa models for entity extraction and intent recognition.
- Weather Information: Get weather updates by querying the OpenWeatherMap API.
- Wikipedia Queries: Retrieve summaries from Wikipedia based on user questions.
- Application Control: Open, close, or launch applications installed on your system.
- Website Navigation: Open websites based on user commands (e.g., Google, YouTube).
- Text-to-Speech (TTS): Convert assistant's responses into speech using pyttsx3.

##Setup
###Requirements:
- Python 3.x
- spacy, torch, pyttsx3, pyaudio, requests, wikipedia-api, whisper, transformers, fuzzywuzzy
- Internet connection for API calls (weather, Wikipedia, etc.)
- Optional: GPU for faster processing (if using Whisper with GPU)

##Installation
git clone https://github.com/your-username/My-Assistant.git
cd My-Assistant

###Download spaCy and the English model:
- python -m spacy download en_core_web_lg

###Set up the necessary environment variables (e.g., your OpenWeatherMap API key):
OpenWeatherMap API key: [Sign up here](https://home.openweathermap.org/users/sign_in)
Add your key in the appropriate variable in the code.

###Run the assistant:
- python assistant.py

##Usage
Once the assistant is running, you can interact with it by speaking commands. The assistant will listen to your voice and execute the following actions based on your commands:
Weather: "What is the weather in Paris?"
Wikipedia: "Tell me about Python programming."
Application Control: "Open Notepad" or "Close Calculator"
Web Navigation: "Open YouTube"
The assistant will respond via text-to-speech with the appropriate answer or action.

##Technologies Used:
- Whisper: For speech recognition.
- spaCy: For entity recognition and NLP tasks.
- RoBERTa: For fine-tuned Named Entity Recognition (NER) to identify app names and other entities.
- pyttsx3: For text-to-speech functionality.
- requests: To fetch weather data from the OpenWeatherMap API.
- Wikipedia-API: To fetch data from Wikipedia.

 ##Contributing
Contributions are welcome! Feel free to fork this repository, open issues, and submit pull requests.


