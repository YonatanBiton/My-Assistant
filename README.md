# Voice-Controlled Personal Assistant
## Version 0.1.0 
This project is a Python-based voice-controlled personal assistant. It utilizes advanced speech recognition, natural language processing, and text-to-speech capabilities to execute various commands such as fetching weather information, searching Wikipedia, opening or closing applications, and more.

## Features

- **Speech Recognition**: Uses OpenAI's Whisper model to transcribe voice commands.
- **Natural Language Processing**: Integrates spaCy for entity extraction and command understanding.
- **Text-to-Speech**: Provides auditory feedback using `pyttsx3`.
- **Weather Updates**: Fetches real-time weather information using the OpenWeatherMap API.
- **Wikipedia Integration**: Retrieves summaries for general knowledge questions.
- **Application Management**: Open, close, and manage apps on your system.
- **Web Browsing**: Opens websites directly from voice commands.
- **Extensibility**: Dynamic database integration for apps and websites.

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip
- SQLite

### Libraries

Install required libraries using pip:

```bash
pip install spacy pyttsx3 numpy pyaudio requests wikipedia-api fuzzywuzzy torch transformers
```

### Additional Setup

1. **Download and Set Up Whisper Model**:
   ```bash
   pip install transformers
   ```

2. **Database Initialization**:
   The project initializes an SQLite database (`assistant_data.db`) for storing application paths and website URLs.

3. **API Keys**:
   - OpenWeatherMap API key: Replace `weather_api_key` with your API key in the script.

4. **spaCy Model**:
   Download the spaCy language model:
   ```bash
   python -m spacy download en_core_web_lg
   ```

## Usage

1. **Run the Assistant**:
   ```bash
   python assistant.py
   ```

2. **Commands**:
   - Weather: "What is the weather in [city]?"
   - Wikipedia: "Tell me about [topic]."
   - Open Apps: "Open [application name]."
   - Close Apps: "Close [application name]."
   - Add New App: "Add [application name]" (Follow prompts to add new apps).
   - Web Browsing: "Open [website name]."

3. **Exit the Assistant**:
   Say "exit" or "goodbye" to close the assistant.

## How It Works

### Speech Recognition
The assistant uses OpenAI's Whisper model to transcribe voice inputs into text. The transcription is processed to extract relevant entities and commands.

### NLP with spaCy
SpaCy's Entity Ruler dynamically adds patterns for recognizing app names, locations, and more. The assistant identifies intents like opening apps, fetching weather, or searching the web.

### Database Integration
SQLite is used to manage whitelisted apps and commonly used websites. The database is initialized automatically if not found.

### Dynamic Execution
The assistant uses subprocesses for opening/closing applications and web browsers for opening websites. It also handles errors gracefully, ensuring a seamless user experience.

## Project Structure

```
.
├── assistant.py          # Main script
├── data_base_init.py     # Database initialization and schema creation
├── requirements.txt      # List of required Python libraries
└── README.md             # Project documentation
```

## Future Improvements

- Automated file path finder
- GUI
- Hybrid text and voice commands for the assistant
- Expand integration with third-party APIs (e.g., Google Calendar, Spotify).

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for improvements.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [spaCy](https://spacy.io/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Wikipedia API](https://pypi.org/project/wikipedia-api/)
