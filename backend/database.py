"""
build the connection to the database to communicate with Postgresql
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
# from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger(__name__)

# read database address from environment variable
DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://")

# define basic class
base = declarative_base()

# create engine (connection)
engine = create_engine(DATABASE_URL,
                       echo=False, # True=print SQL-Command in consol out for Debug
                       pool_pre_ping=True) # test connection before use

# Session-Factory (each request get its own session)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def connect_database():
    """
    test whether database is connected or not
    return: true or false
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("database connected successfully.")
        return True
    except Exception as e:
        logger.error(f"database connected failed: {e}")
        return False

def create_table():
    """
    create table if not exists
    :return:
    """
    base.metadata.create_all(bind=engine)
    logger.info("Table created/checked successfully.")

def get_db():
    """
    FastAPI Dependency: give a database connection and close the connection in the end automatically
    :return:
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()