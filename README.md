
# Virtual Assistant with Speech Recognition and NLP

This project is a Python-based virtual assistant that utilizes advanced speech recognition, natural language processing (NLP), and multiple APIs to provide real-time functionalities like weather updates, Wikipedia queries, and application control. This assistant can open and close applications, search the web, and respond to user queries using text-to-speech.

## Features

- **Speech Recognition**: Uses `whisper` for high-accuracy speech-to-text conversion.
- **Natural Language Processing (NLP)**: Utilizes `spaCy` to extract entities such as cities, apps, and websites.
- **Weather Updates**: Fetches live weather data for any city via the OpenWeatherMap API.
- **Wikipedia Integration**: Retrieves answers to user queries from Wikipedia.
- **Application Control**: Open and close applications based on voice commands.
- **Website Access**: Opens websites based on spoken instructions.
- **Text-to-Speech**: Uses `pyttsx3` to read responses aloud.
- **Data Persistence**: Stores whitelisted applications, websites, and common queries in an SQLite database.

## Requirements

Ensure you have the following Python packages installed:

- `spaCy` for NLP: `pip install spacy`
- `pyttsx3` for Text-to-Speech: `pip install pyttsx3`
- `whisper` for Speech Recognition: `pip install whisper`
- `requests` for HTTP requests to weather API: `pip install requests`
- `wikipedia-api` for Wikipedia queries: `pip install wikipedia-api`
- `rapidfuzz` for fuzzy matching in NLP tasks: `pip install rapidfuzz`
- `sounddevice` for audio recording: `pip install sounddevice`
- SQLite (comes pre-installed with Python)

Additionally, you will need to sign up for an API key from OpenWeatherMap:

1. Go to [OpenWeatherMap](https://openweathermap.org/), create an account, and generate a free API key.
2. Replace the `weather_api_key` in your script with the obtained API key.

## Installation

Follow these steps to get your local copy of the project up and running:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/virtual-assistant.git
   cd virtual-assistant
   ```

2. **Install required dependencies**:
   Install all required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```


3. **Set your API Key**:
   Update the `weather_api_key` variable in the script to your OpenWeatherMap API key.

## Usage

1. **Run the Assistant**: Start the assistant by executing:
   ```bash
   python assistant_main.py
   ```

2. **Interact with the Assistant**: You can issue commands using voice or text input. Some example commands:
   - "What is the weather in New York?"
   - "Open Google Chrome"
   - "Search for Python tutorials on Wikipedia"
   - "Close Spotify"
   - "Open YouTube"

3. **Control Applications**: The assistant can open and close applications on your system:
   - "Open [application_name]"
   - "Close [application_name]"

4. **Search Wikipedia**: You can ask the assistant to look up topics on Wikipedia:
   - "What is the capital of Japan?"

5. **Weather Queries**: The assistant fetches weather details using OpenWeatherMap:
   - "What is the weather like in [city]?"

6. **Opening Websites**: The assistant can open websites based on voice commands:
   - "Open [website_name]"


## Project Structure

```
.
├── assistant.py          # Main script
├── data_base_init.py     # Database initialization and schema creation
├── requirements.txt      # List of required Python libraries
└── README.md             # Project documentation
```


## Database Structure

The project uses SQLite for storing common application names, websites, and question starters. The database contains three main tables:

- **whitelist_apps**: Stores the names and paths of user applications that can be opened or closed via voice commands.
- **common_websites**: Stores common websites that can be opened by voice commands.
- **question_starters**: Stores common phrases that are used to identify user queries (e.g., "What is, ..." etc.).

To view or modify the data, you can use an SQLite database browser or access it programmatically via Python.

## Example Workflow

1. **Start the assistant**: Run the assistant, and it will begin listening for commands.
2. **Speak a command**: For example, "Open Google Chrome."
3. **Assistant responds**: The assistant will open the Google Chrome application and respond, "Opening Google Chrome."
4. **Ask for weather**: You can ask, "What is the weather in Paris?" and it will fetch real-time data from OpenWeatherMap and read it aloud.
5. **Search Wikipedia**: You can ask, "Who is Barack Obama?," and the assistant will fetch a summary from Wikipedia.

## Contributing

We welcome contributions to this project! Here’s how you can help:

- **Report Bugs**: If you find any bugs or issues, please create a GitHub issue.
- **Suggest Features**: If you have any suggestions for new features or improvements, feel free to open an issue or submit a pull request.
- **Code Improvements**: If you can improve the code or add new functionalities, feel free to fork the repo and create a pull request.

## Future Improvements

- Automated file path finder
- GUI
- Hybrid text and voice commands for the assistant
- Expand integration with third-party APIs (e.g., Google Calendar, Spotify).

### Setting up a Development Environment

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-username/virtual-assistant.git
   cd virtual-assistant
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a new branch for your feature:
   ```bash
   git checkout -b new-feature
   ```

4. After making changes, commit them:
   ```bash
   git commit -am "Add feature X"
   ```

5. Push your changes and submit a pull request:
   ```bash
   git push origin new-feature
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- **Whisper**: For robust and accurate speech recognition.
- **spaCy**: For advanced natural language processing.
- **pyttsx3**: For text-to-speech functionality.
- **OpenWeatherMap API**: For fetching real-time weather information.
- **Wikipedia-API**: For Wikipedia integration.

---
