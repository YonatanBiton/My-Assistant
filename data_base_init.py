import sqlite3

def create_connection(db_name):
    """Create a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        print(f"Connected to the database {db_name}")
        return conn
    except sqlite3.Error as e:
        print(f"Error occurred while connecting to database: {e}")
        return None

def create_tables(cursor):
    """Create tables for question starters, whitelist apps, and common websites."""
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS question_starters (
            id INTEGER PRIMARY KEY,
            category TEXT,
            phrase TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS whitelist_apps (
            id INTEGER PRIMARY KEY,
            app_name TEXT UNIQUE,
            app_path TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS common_websites (
            id INTEGER PRIMARY KEY,
            category TEXT,
            website_name TEXT,
            url TEXT
        )
        ''')
        print("Tables created successfully or already exist.")
    except sqlite3.Error as e:
        print(f"Error creating tables: {e}")

def insert_question_starters(cursor, question_starters):
    """Insert question starters into the database."""
    try:
        for category, phrases in question_starters.items():
            for phrase in phrases:
                cursor.execute('''
                INSERT INTO question_starters (category, phrase)
                VALUES (?, ?)
                ''', (category, phrase))
        print("Question starters inserted successfully.")
    except sqlite3.Error as e:
        print(f"Error inserting question starters: {e}")

def insert_whitelist_apps(cursor, whitelist_apps):
    """Insert whitelist apps into the database."""
    try:
        for app_name, app_path in whitelist_apps.items():
            cursor.execute('''
            INSERT INTO whitelist_apps (app_name, app_path)
            VALUES (?, ?)
            ''', (app_name, app_path))
        print("Whitelist apps inserted successfully.")
    except sqlite3.Error as e:
        print(f"Error inserting whitelist apps: {e}")

def insert_common_websites(cursor, common_websites):
    """Insert common websites into the database."""
    try:
        for category, websites in common_websites.items():
            for website_name, url in websites.items():
                cursor.execute('''
                INSERT INTO common_websites (category, website_name, url)
                VALUES (?, ?, ?)
                ''', (category, website_name, url))
        print("Common websites inserted successfully.")
    except sqlite3.Error as e:
        print(f"Error inserting common websites: {e}")

def main():
    """Main function to run the operations."""
    # Define the SQLite database filename
    db_filename = 'assistant_data.db'

    # Define data to insert
    question_starters = {
        "who": ["who is", "who was", "who are", "who were"],
        "what": ["what is", "what was", "what are", "what were"],
        "where": ["where is", "where was", "where are", "where were"],
        "when": ["when is", "when was", "when are", "when were"],
        "why": ["why is", "why was", "why are", "why were"],
        "how": ["how is", "how was", "how are", "how were"]
    }

    whitelist_apps = {
        "notepad": r"C:\Windows\System32\notepad.exe",
        "calculator": r"C:\Windows\System32\calc.exe",
        "snippingtool": r"C:\Windows\System32\SnippingTool.exe",
        "paint": r"C:\Windows\System32\mspaint.exe",
    }

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

    # Establish a database connection and create the tables
    conn = create_connection(db_filename)
    if conn is None:
        return  # Exit if connection failed

    cursor = conn.cursor()

    # Create tables in the database
    create_tables(cursor)

    # Insert data into the database
    insert_question_starters(cursor, question_starters)
    insert_whitelist_apps(cursor, whitelist_apps)
    insert_common_websites(cursor, common_websites)

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
    print("Database operations completed successfully.")

if __name__ == "__main__":
    main()
