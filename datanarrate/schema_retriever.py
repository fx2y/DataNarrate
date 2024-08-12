import json
import logging
from typing import Dict, Any, Optional, List

import urllib3
from elasticsearch import Elasticsearch
from langchain_core.pydantic_v1 import BaseModel, Field
from sqlalchemy import create_engine, inspect

from config import config

# Disable warnings about insecure connections
urllib3.disable_warnings()


class ColumnInfo(BaseModel):
    name: str = Field(description="Name of the column")
    data_type: str = Field(description="Data type of the column")
    is_nullable: bool = Field(description="Whether the column can be null")
    is_primary_key: bool = Field(description="Whether the column is a primary key")


class TableSchema(BaseModel):
    name: str = Field(description="Name of the table")
    columns: List[ColumnInfo] = Field(description="List of columns in the table")


class MySQLSchema(BaseModel):
    tables: List[TableSchema] = Field(description="List of tables in the MySQL database")


class ElasticsearchIndex(BaseModel):
    name: str = Field(description="Name of the Elasticsearch index")
    mappings: Dict[str, Any] = Field(description="Mappings of the Elasticsearch index")


class ElasticsearchSchema(BaseModel):
    indices: List[ElasticsearchIndex] = Field(description="List of Elasticsearch indices")


class UnifiedSchema(BaseModel):
    mysql: MySQLSchema = Field(description="MySQL schema information")
    elasticsearch: ElasticsearchSchema = Field(description="Elasticsearch schema information")


class SchemaRetriever:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.engine = None
        self.es_client = None

    def _get_engine(self):
        if not self.engine:
            self.engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
        return self.engine

    def _get_es_client(self):
        if not self.es_client:
            self.es_client = Elasticsearch(
                hosts=[config.ELASTICSEARCH_HOST],
                verify_certs=False,
                basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD)
            )
        return self.es_client

    def retrieve_mysql_schema(self, database: str) -> MySQLSchema:
        try:
            self.logger.info(f"Retrieving MySQL schema for database: {database}")
            engine = self._get_engine()
            inspector = inspect(engine)

            schema = MySQLSchema(tables=[])
            for table_name in inspector.get_table_names():
                columns = []
                for column in inspector.get_columns(table_name):
                    columns.append(ColumnInfo(
                        name=column['name'],
                        data_type=str(column['type']),
                        is_nullable=column['nullable'],
                        is_primary_key=column.get('primary_key', False)
                    ))
                schema.tables.append(TableSchema(name=table_name, columns=columns))

            self.logger.info(f"Successfully retrieved MySQL schema for database: {database}")
            return schema
        except Exception as e:
            self.logger.error(f"Error retrieving MySQL schema: {e}", exc_info=True)
            return MySQLSchema(tables=[])

    def retrieve_elasticsearch_schema(self, index_pattern: str) -> ElasticsearchSchema:
        try:
            self.logger.info(f"Retrieving Elasticsearch schema for index pattern: {index_pattern}")
            es = self._get_es_client()
            indices = es.indices.get(index=index_pattern)

            schema = ElasticsearchSchema(indices=[])
            for index_name, index_info in indices.items():
                index_schema = ElasticsearchIndex(
                    name=index_name,
                    mappings=self._process_es_mappings(index_info['mappings'])
                )
                schema.indices.append(index_schema)

            self.logger.info(f"Successfully retrieved Elasticsearch schema for index pattern: {index_pattern}")
            return schema
        except Exception as e:
            self.logger.error(f"Error retrieving Elasticsearch schema: {e}", exc_info=True)
            return ElasticsearchSchema(indices=[])

    def _process_es_mappings(self, mappings: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        processed = {}
        for field, properties in mappings.get("properties", {}).items():
            field_type = properties.get("type", "object")
            full_field_name = f"{prefix}{field}"
            if field_type == "nested":
                processed[full_field_name] = {
                    "type": "nested",
                    "properties": self._process_es_mappings(properties, f"{full_field_name}.")
                }
            elif field_type == "object":
                processed[full_field_name] = self._process_es_mappings(properties, f"{full_field_name}.")
            else:
                processed[full_field_name] = {"type": field_type}
        return processed

    def retrieve_unified_schema(self, mysql_database: str, es_index_pattern: str) -> UnifiedSchema:
        mysql_schema = self.retrieve_mysql_schema(mysql_database)
        es_schema = self.retrieve_elasticsearch_schema(es_index_pattern)

        unified_schema = UnifiedSchema(
            mysql=mysql_schema,
            elasticsearch=es_schema
        )
        return unified_schema

    def compress_schema(self, unified_schema: UnifiedSchema) -> Dict[str, Any]:
        """
        Create a compressed version of the unified schema.
        """
        compressed = {
            "mysql": self._compress_mysql_schema(unified_schema.mysql),
            "elasticsearch": self._compress_elasticsearch_schema(unified_schema.elasticsearch)
        }
        return compressed

    def _compress_mysql_schema(self, mysql_schema: MySQLSchema) -> Dict[str, Any]:
        compressed = {}
        for table in mysql_schema.tables:
            compressed[table.name] = [
                f"{col.name}:{col.data_type[:3]}{'?' if col.is_nullable else ''}{'*' if col.is_primary_key else ''}"
                for col in table.columns
            ]
        return compressed

    def _compress_elasticsearch_schema(self, es_schema: ElasticsearchSchema) -> Dict[str, Any]:
        compressed = {}
        for index in es_schema.indices:
            compressed[index.name] = self._compress_es_mappings(index.mappings)
        return compressed

    def _compress_es_mappings(self, mappings: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        compressed = {}
        for field, properties in mappings.items():
            full_field_name = f"{prefix}{field}"
            if 'type' in properties:
                field_type = properties['type']
                if field_type == "nested":
                    compressed[full_field_name] = "nes"
                    nested_fields = self._compress_es_mappings(properties.get("properties", {}), f"{full_field_name}.")
                    compressed.update(nested_fields)
                else:
                    compressed[full_field_name] = field_type[:3]
            elif isinstance(properties, dict) and any(isinstance(v, dict) for v in properties.values()):
                # This is an implicitly nested object
                compressed[full_field_name] = "obj"
                nested_fields = self._compress_es_mappings(properties, f"{full_field_name}.")
                compressed.update(nested_fields)
            else:
                # This might be a field with additional properties, default to object
                compressed[full_field_name] = "obj"
        return compressed

    def close_connections(self):
        if self.engine:
            self.engine.dispose()
        if self.es_client:
            self.es_client.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=config.LOG_LEVEL)

    # Initialize SchemaRetriever
    schema_retriever = SchemaRetriever()

    try:
        # Retrieve unified schema
        unified_schema = schema_retriever.retrieve_unified_schema(
            config.MYSQL_DATABASE,
            config.ELASTICSEARCH_INDEX_PATTERN
        )
        print("Unified Schema:")
        print(unified_schema.json(indent=2))

        # Compress the schema
        compressed_schema = schema_retriever.compress_schema(unified_schema)
        print("\nCompressed Schema:")
        print(json.dumps(compressed_schema, indent=2))
    finally:
        schema_retriever.close_connections()
