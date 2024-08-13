import logging
from typing import Union, Dict, Any, Optional

from elasticsearch_dsl import Search
from sqlparse import parse as sql_parse
from sqlparse.sql import Statement as SQLStatement

from config import config


class QueryValidator:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_query(self, query: Union[str, Dict[str, Any]], query_type: str) -> bool:
        """
        Validate the given query based on its type.

        :param query: The query to validate (SQL string or Elasticsearch query dict)
        :param query_type: The type of query ('sql' or 'elasticsearch')
        :return: True if the query is valid, False otherwise
        """
        if query_type.lower() == 'sql':
            return self.validate_sql_query(query)
        elif query_type.lower() == 'elasticsearch':
            return self.validate_elasticsearch_query(query)
        else:
            self.logger.error(f"Unsupported query type: {query_type}")
            return False

    def validate_sql_query(self, query: str) -> bool:
        """
        Validate the SQL query.

        :param query: The SQL query string to validate
        :return: True if the query is valid, False otherwise
        """
        try:
            self.logger.info("Validating SQL query")
            parsed = sql_parse(query)
            if not parsed:
                self.logger.error("Failed to parse SQL query")
                return False

            statement = parsed[0]
            if not isinstance(statement, SQLStatement):
                self.logger.error("Invalid SQL statement")
                return False

            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'TRUNCATE', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            if any(keyword in statement.tokens[0].value.upper() for keyword in dangerous_keywords):
                self.logger.error(f"Dangerous operation detected in SQL query: {statement.tokens[0].value}")
                return False

            self.logger.info("SQL query validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating SQL query: {e}", exc_info=True)
            return False

    def validate_elasticsearch_query(self, query: Dict[str, Any]) -> bool:
        """
        Validate the Elasticsearch query.

        :param query: The Elasticsearch query dict to validate
        :return: True if the query is valid, False otherwise
        """
        try:
            self.logger.info("Validating Elasticsearch query")
            # Use elasticsearch_dsl to validate the query structure
            search = Search.from_dict(query)

            # Check for dangerous operations
            if 'script' in str(search.to_dict()):
                self.logger.error("Dangerous operation detected in Elasticsearch query: script execution")
                return False

            self.logger.info("Elasticsearch query validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating Elasticsearch query: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=config.LOG_LEVEL)

    # Initialize QueryValidator
    validator = QueryValidator()

    # Test SQL query validation
    sql_query = "SELECT * FROM users WHERE age > 18"
    print(f"SQL Query Valid: {validator.validate_query(sql_query, 'sql')}")

    # Test dangerous SQL query
    dangerous_sql_query = "DROP TABLE users"
    print(f"Dangerous SQL Query Valid: {validator.validate_query(dangerous_sql_query, 'sql')}")

    # Test Elasticsearch query validation
    es_query = {
        "query": {
            "match": {
                "title": "python"
            }
        }
    }
    print(f"Elasticsearch Query Valid: {validator.validate_query(es_query, 'elasticsearch')}")

    # Test dangerous Elasticsearch query
    dangerous_es_query = {
        "query": {
            "script": {
                "script": {
                    "source": "System.exit(0)"
                }
            }
        }
    }
    print(f"Dangerous Elasticsearch Query Valid: {validator.validate_query(dangerous_es_query, 'elasticsearch')}")
