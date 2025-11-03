"""
SQL Database Module
===================

This module provides a production-grade SQLAlchemy wrapper for database operations
in Neurosurfer. It's designed for use with SQL agents and tools, providing schema
introspection, query execution, and metadata caching.

Key Features:
    - SQLAlchemy-based database abstraction
    - Schema introspection and caching
    - Table filtering (include/ignore lists)
    - Sample data retrieval for context
    - View support
    - Metadata persistence for performance
    - Safe query execution with error handling

The SQLDatabase class is used by SQL agents to understand database schemas
and execute queries safely.

Example:
    >>> from neurosurfer.db.sql_database import SQLDatabase
    >>> 
    >>> # Connect to database
    >>> db = SQLDatabase(
    ...     database_uri="postgresql://user:pass@localhost/mydb",
    ...     include_tables=["users", "orders"],
    ...     sample_rows_in_table_info=3
    ... )
    >>> 
    >>> # Get schema information
    >>> schema = db.get_table_info()
    >>> 
    >>> # Execute query
    >>> result = db.run("SELECT * FROM users LIMIT 5")
"""
import os
import pickle
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Any, Union, Iterable
from urllib.parse import quote_plus

import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import URL, Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType



class SQLDatabase:
    """
    Production-grade SQLAlchemy database wrapper.
    
    This class provides a high-level interface for database operations,
    including schema introspection, query execution, and metadata caching.
    It's designed for use with SQL agents and tools.
    
    Features:
        - Automatic schema introspection
        - Table filtering (include/ignore)
        - Sample data for context
        - Metadata caching for performance
        - View support
        - Safe query execution
    
    Attributes:
        _engine (Engine): SQLAlchemy engine
        _schema (Optional[str]): Database schema name
        _inspector: SQLAlchemy inspector
        _all_tables (set): All available tables
        _usable_tables (set): Filtered tables based on include/ignore
    
    Example:
        >>> db = SQLDatabase(
        ...     database_uri="sqlite:///mydb.db",
        ...     include_tables=["users", "products"],
        ...     sample_rows_in_table_info=3
        ... )
        >>> 
        >>> # Get schema
        >>> schema_info = db.get_table_info()
        >>> 
        >>> # Execute query
        >>> results = db.run("SELECT COUNT(*) FROM users")
        >>> 
        >>> # Get table names
        >>> tables = db.get_usable_table_names()
    """
    
    def __init__(
        self,
        database_uri: Union[str, URL],
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: bool = False,
        max_string_length: int = 300,
        lazy_table_reflection: bool = False,
        # NEW:
        metadata_cache_dir: Optional[Union[str, Path]] = None,
        force_refresh: bool = False,
        cache_ttl_seconds: Optional[int] = None,  # e.g., 86400 for 1 day
    ):
        """Create engine from database URI."""
        self._engine = create_engine(database_uri)
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)

        # including view support by adding the views as well as tables to the all
        # tables list if view_support is True
        self._all_tables = set(
            self._inspector.get_table_names(schema=schema)
            + (self._inspector.get_view_names(schema=schema) if view_support else [])
        )

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

        self._max_string_length = max_string_length
        self._view_support = view_support

        # ---- CACHED METADATA LOGIC ----
        self._metadata = metadata or MetaData()
        self._metadata_cache_dir = Path(
            metadata_cache_dir or Path.home() / ".cache" / "sqlalchemy_metadata"
        )
        self._metadata_cache_dir.mkdir(parents=True, exist_ok=True)

        if not lazy_table_reflection:
            cache_path = self._metadata_cache_path()
            loaded = False

            if cache_path and not force_refresh:
                loaded = self._maybe_load_metadata_cache(cache_path, cache_ttl_seconds)

            if not loaded:
                self._build_and_cache_metadata(cache_path)


        # self._metadata = metadata or MetaData()
        # if not lazy_table_reflection:
        #     # including view support if view_support = true
        #     self._metadata.reflect(
        #         views=view_support,
        #         bind=self._engine,
        #         only=list(self._usable_tables),
        #         schema=self._schema,
        #     )
    
    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.
        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        to_reflect = set(all_table_names) - set(metadata_table_names)
        if to_reflect:
            self._metadata.reflect(
                views=self._view_support,
                bind=self._engine,
                only=list(to_reflect),
                schema=self._schema,
            )

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # Ignore JSON datatyped columns
            for k, v in table.columns.items():  # AttributeError: items in sqlalchemy v1
                if type(v.type) is NullType:
                    table._columns.remove(v)

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            has_extra_info = (
                self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                table_info += f"\n{self._get_sample_rows(table)}\n"
            if has_extra_info:
                table_info += "*/"
            tables.append(table_info)
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"

    def _get_sample_rows(self, table: Table) -> str:
        # build the select command
        command = select(table).limit(self._sample_rows_in_table_info)
        # save the columns in string format
        columns_str = "\t".join([col.name for col in table.columns])
        try:
            # get the sample rows
            with self._engine.connect() as connection:
                sample_rows_result = connection.execute(command)  # type: ignore
                # shorten values in the sample rows
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
                )
            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            sample_rows_str = ""
        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )

    @staticmethod
    def build_connection_string(
        server: str,
        database: str,
        username: str,
        password: str,
        driver: str = "mssql+pyodbc",  # e.g., "postgresql", "mysql", "mssql+pyodbc"
        port: Optional[str] = None,
        odbc_driver: Optional[str] = None  # Only needed for ODBC/SQL Server
    ) -> str:
        safe_password = quote_plus(password)

        if driver == "mssql+pyodbc":
            if not odbc_driver:
                raise ValueError("ODBC driver name is required for mssql+pyodbc")
            driver_param = quote_plus(odbc_driver)
            port = port or "1433"
            return f"{driver}://{username}:{safe_password}@{server}:{port}/{database}?driver={driver_param}"

        # Default ports for common databases
        if driver == "postgresql":
            port = port or "5432"
        elif driver == "mysql":
            port = port or "3306"

        return f"{driver}://{username}:{safe_password}@{server}:{port}/{database}"


    # ----------------------
    # Caching helpers
    # ----------------------
    def _metadata_cache_path(self) -> Optional[Path]:
        """
        Build a stable cache path that depends on:
          - DB dialect + database name (safe-ish; we avoid dumping full URI with secrets)
          - schema
          - view_support flag
          - the sorted list of usable tables
        Returns None for in-memory SQLite, where caching is pointless.
        """
        url = str(self._engine.url)
        # Skip in-memory SQLite
        if url.startswith("sqlite:///:memory:"):
            return None

        # Derive a friendly identity (don’t store credentials):
        # Dialect + database name (path for sqlite file)
        dialect = self._engine.url.get_dialect().name
        database_id = (
            self._engine.url.database or ""  # may be file path for sqlite
        )
        schema = self._schema or ""

        key_payload = {
            "dialect": dialect,
            "database_id": database_id,
            "schema": schema,
            "view_support": bool(self._view_support),
            "tables": sorted(list(self._usable_tables)),
        }
        key_str = repr(key_payload).encode("utf-8")
        digest = hashlib.sha256(key_str).hexdigest()[:16]

        fname = f"meta_{dialect}_{os.path.basename(database_id) or 'default'}_{schema or 'default'}_{'views' if self._view_support else 'tables'}_{digest}.pkl"
        return self._metadata_cache_dir / fname

    def _maybe_load_metadata_cache(
        self, cache_path: Optional[Path], cache_ttl_seconds: Optional[int]
    ) -> bool:
        """Try to load cached MetaData if present (and not expired if TTL is set)."""
        if not cache_path or not cache_path.exists():
            return False

        if cache_ttl_seconds is not None:
            age = time.time() - cache_path.stat().st_mtime
            if age > cache_ttl_seconds:
                return False  # too old

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, MetaData):
                self._metadata = cached
                return True
        except Exception:
            return False
        return False

    def _build_and_cache_metadata(self, cache_path: Optional[Path]) -> None:
        """Reflect metadata from DB and cache it if a path is available."""
        md = MetaData()
        md.reflect(
            views=self._view_support,
            bind=self._engine,
            only=list(self._usable_tables),
            schema=self._schema,
        )
        self._metadata = md
        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(md, f)
            except Exception:
                # best effort; ignore cache write failures
                pass