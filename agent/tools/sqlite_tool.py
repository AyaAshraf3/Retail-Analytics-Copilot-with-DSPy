"""
SQLite Tool for Northwind Database Access 
- Schema introspection via PRAGMA
- Query execution with error capture
- Validation of SQL statements
"""

import sqlite3
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SQLiteTool:
    """SQLite database tool for Northwind queries."""
    
    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path
        self.connection = None
        self.connect()

    def connect(self):
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_schema(self) -> str:
        """
        Retrieve database schema via PRAGMA statements.
        Returns formatted schema description for DSPy context.
        """
        try:
            cursor = self.connection.cursor()
            
            # Get all tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()
            
            schema_info = "SQLite Northwind Schema:\n\n"
            
            for table in tables:
                table_name = table[0]
                schema_info += f"Table: [{table_name}]\n"
                
                try:
                    # Get columns for each table
                    cursor.execute(f"PRAGMA table_info([{table_name}])")
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2]
                        schema_info += f" - {col_name}: {col_type}\n"
                
                except Exception as e:
                    logger.warning(f"Could not get info for table {table_name}: {e}")
                    continue
                
                schema_info += "\n"
            
            return schema_info
        
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            raise

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute SQL query safely.
        Returns dict with rows, columns, or error message.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Convert rows to list of dicts
            result_rows = [dict(row) for row in rows]
            
            return {
                "rows": result_rows,
                "columns": columns,
                "error": None,
                "row_count": len(result_rows)
            }
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query execution error: {error_msg}")
            return {
                "rows": [],
                "columns": [],
                "error": error_msg,
                "row_count": 0
            }

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query by actually executing it.
        Returns (is_valid, error_message).
        
        Checks:
        1. Starts with SELECT
        2. Has FROM clause
        3. Has valid table names
        4. Matching parentheses
        5. Actually executes in test transaction
        """
        if not query or not isinstance(query, str):
            return False, "Query is empty or invalid type"
        
        query_upper = query.upper().strip()
        
        # Check 1: Must start with SELECT
        if not query_upper.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Check 2: Must have FROM clause
        if ' FROM ' not in query_upper:
            return False, "Query missing FROM clause"
        
        # Check 3: Must have valid table names (case-insensitive)
        valid_tables = ['ORDER DETAILS', 'ORDERS', 'PRODUCTS', 'CUSTOMERS', 
                        'CATEGORIES', 'SUPPLIERS', 'SHIPPERS', 'EMPLOYEES']
        has_table = any(table in query_upper for table in valid_tables)
        if not has_table:
            return False, "No valid Northwind table found"
        
        # Check 4: Matching parentheses
        if query_upper.count('(') != query_upper.count(')'):
            return False, "Mismatched parentheses"
        
        # Check 5: Not placeholder query
        placeholder_queries = [
            'SELECT 1', 'SELECT 1;',
            'SELECT COUNT(*) AS RESULT FROM ORDERS',
            'SELECT COUNT(*) FROM ORDERS'
        ]
        if query.strip().upper() in placeholder_queries:
            return False, "Query is a placeholder (fallback)"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("BEGIN")
            try:
                # Execute the query
                cursor.execute(query)
                # Fetch at least one row to ensure it's valid
                cursor.fetchone()
                # Rollback the test transaction
                cursor.execute("ROLLBACK")
                
                logger.info(f"✓ Query validated: {query[:80]}")
                return True, None
            
            except Exception as exec_error:
                # Execution failed - rollback and return error
                try:
                    cursor.execute("ROLLBACK")
                except:
                    pass  
                
                error_msg = str(exec_error)
                logger.warning(f"Query execution failed: {error_msg}")
                return False, error_msg
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Validation error: {error_msg}")
            return False, error_msg

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
