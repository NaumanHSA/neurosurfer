import json
import os
import re
import logging
from typing import Dict, Optional, Generator, Any
from tqdm import tqdm
from ..models.chat_models.base import BaseModel
from .sql_database import SQLDatabase

SCHEMA_SUMMARIZATION_PROMPT = {
    "system_prompt": """You are a master of SQL. 
Your sole task is to analyze and summarize SQL table schemas in a concise and structured manner.
Do not include any usage examples, assumptions, or extra explanation outside the specified structure.""",

    "prompt": """Given the SQL schema of the `{table_name}` table, explain the following points. You must output josn object with the following structure:

```json
{{
    "table_name": "The name of the table.",
    "summary": "A brief description of what data the `{table_name}` table stores and its role in the system."
}}

```        
Respond only with this json object — nothing else. Do not make the headings in bold.

Here is the schema:
{schema}"""
}


class SQLSchemaStore:
    def __init__(
        self,
        db_uri: str,
        llm: Optional[BaseModel] = None,
        sample_rows_in_table_info: int = 3,
        storage_path: Optional[str] = None,
        logger: Optional[logging.Logger] = logging.getLogger(),
    ):
        self.llm = llm
        self.logger = logger    
        self.db_uri = db_uri
        self.db = SQLDatabase(
            self.db_uri,
            sample_rows_in_table_info=sample_rows_in_table_info,
            view_support=True,
            force_refresh=False
        )
        self.storage_path = os.path.join(storage_path or "./", f"sql_schema_store_{self.get_db_name()}.json")
        self.store: Dict[str, str] = self.load_from_file() if os.path.exists(self.storage_path) else {}
        self.logger.info(f"[SQLStore] Loaded {len(self.store)} schema summaries.")

    def train(self, summarize: bool = False, force: bool = False) -> Generator:
        """
        Trains the schema summarizer by extracting schema data and optionally generating LLM summaries.
        summarize:
            - False: Store raw schema only.
            - True: Add LLM-generated short summary with schema.
        force:
            - If True, flushes existing storage and retrains all tables.
        """
        if summarize and self.llm is None:
            raise ValueError("LLM is required for summarize level 'short' or 'detailed'.")
        if self.store and not force:
            self.logger.info("[SQLStore] Already trained. Use `force=True` to retrain.")
            return
        if force:
            self.store.clear()
            self.logger.info("[SQLStore] Force retrain enabled. Previous summaries flushed.")
        schemas = self.get_all_table_schemas()
        self.logger.info(f"[SQLStore] Training started on {len(schemas)} tables...")
        processed_tables = 0
        for table, schema in tqdm(schemas.items()):
            if summarize:
                schema_dict = self.summarize_schema__(table, schema)
                schema_dict["schema"] = schema   # add table schema and example rows
            self.store[table] = schema_dict
            processed_tables += 1
            yield processed_tables
        self.save_to_file()
        self.logger.info(f"[SQLStore] Training completed. Stored {len(self.store)} entries.")

    def summarize_schema__(self, table_name: str, schema: str) -> Dict[str, Any]:
        prompt = SCHEMA_SUMMARIZATION_PROMPT["prompt"].format(table_name=table_name, schema=schema)
        system_prompt = SCHEMA_SUMMARIZATION_PROMPT["system_prompt"]
        response = self.llm.ask(
            user_prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_new_tokens=3000,
            stream=False,
        )["choices"][0]["message"]["content"]
        response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(response)

    def get_all_table_schemas(self) -> Dict[str, str]:
        tables = self.db.get_usable_table_names()
        return {t: self.db.get_table_info([t]) for t in tables}

    def save_to_file(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2)

    def load_from_file(self) -> Dict[str, str]:
        with open(self.storage_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_table_data(self, table: str) -> Optional[str]:
        return self.store.get(table)

    def get_db_name(self) -> str:
        match = re.search(r"/([^/?]+)(?:\?|$)", self.db_uri)
        return match.group(1) if match else None

    def get_tables_count(self) -> int:
        return len(self.db.get_usable_table_names())