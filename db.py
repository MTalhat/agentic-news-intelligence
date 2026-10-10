import os
from contextlib import contextmanager
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
pool = ConnectionPool(conninfo=DB_URL, min_size=2, max_size=10, open=True)

@contextmanager
def get_db():
    with pool.connection() as conn:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
