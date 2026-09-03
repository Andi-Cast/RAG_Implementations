import os
import psycopg
from pgvector.psycopg import register_vector


DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn