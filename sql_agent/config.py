import requests
from langchain_community.utilities import SQLDatabase


def download_database(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Database downloaded and saved as {filename}")
    else:
        print(f"Failed to download the database. Status code: {response.status_code}")


def initialize_database(db_path):
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")
