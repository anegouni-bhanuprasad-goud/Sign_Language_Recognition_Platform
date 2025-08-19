from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker, declarative_base

database_url = ""

engine =create_engine(url=database_url, echo=False)

SessionLocal = sessionmaker(bind=engine, autoflush=False)

Base = declarative_base()