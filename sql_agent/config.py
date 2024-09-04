from typing import Tuple

import requests
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel


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


def initialize_mysql_database(
        host: str,
        user: str,
        password: str,
        database: str,
        llm: BaseLanguageModel
) -> Tuple[SQLDatabase, SQLDatabaseToolkit]:
    """
    Initialize a MySQL database connection and create a SQLDatabaseToolkit.

    Args:
        host (str): The MySQL host address.
        user (str): The MySQL user name.
        password (str): The MySQL password.
        database (str): The name of the database to connect to.
        llm (BaseLanguageModel): The language model to use for the toolkit.

    Returns:
        Tuple[SQLDatabase, SQLDatabaseToolkit]: A tuple containing the SQLDatabase instance
        and the SQLDatabaseToolkit.

    Raises:
        Exception: If there's an error connecting to the database.
    """
    try:
        # Construct the MySQL connection string
        connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"

        # Create the SQLDatabase instance
        db = SQLDatabase.from_uri(connection_string)

        # Create the SQLDatabaseToolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        return db, toolkit
    except Exception as e:
        raise Exception(f"Failed to initialize MySQL database: {str(e)}")
