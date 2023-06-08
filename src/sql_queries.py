import pymysql
import sqlalchemy as alch # python -m pip install --upgrade 'sqlalchemy<2.0'

from getpass import getpass
import pandas as pd
from sqlalchemy import create_engine, text
import time
from dotenv import load_dotenv
import os

def connect(df):
    load_dotenv()
    password = os.getenv("password")
    
    dbName = "idealista"
    connectionData=f"mysql+pymysql://root:{password}@localhost/{dbName}"
    engine = alch.create_engine(connectionData)

    
    # Loading to the new info
    table = "idealista"
    df.to_sql(table, con=engine, if_exists='replace', index=False)
    
    return engine

def average_district(engine):
    drop = "DROP VIEW IF EXISTS average_district;"
    select = "SELECT * FROM average_district"
    query = """
    CREATE VIEW average_district AS
    SELECT neighbourhood, average_price, ((average_price - total_avg) / total_avg) * 100 AS percentage_vs_Barcelona
    FROM (
        SELECT neighbourhood, round(AVG(price),0) AS average_price, (SELECT AVG(price) FROM idealista) AS total_avg
        FROM idealista
        GROUP BY neighbourhood
        ORDER by average_price DESC
    ) subquery;
    """
    drop_view = text(drop)
    create_view = text(query)

    with engine.connect() as connection:
        connection.execute(drop_view)
        connection.execute(create_view)

    return pd.read_sql_query(select, engine)


def parking(engine):
    drop = "DROP VIEW IF EXISTS parking;"
    select= "SELECT * from parking"
    query = """
    
    CREATE VIEW parking AS
    SELECT parking, round(avg(price)) as average_price
    FROM idealista
    GROUP BY parking
    ;



    """
    drop_view = text(drop)
    create_view = text(query)



    with engine.connect() as connection:
        connection.execute(drop_view)
        connection.execute(create_view)

    return pd.read_sql_query(select, engine)


def bedrooms(engine):
    drop= "DROP VIEW IF EXISTS bedrooms;"
    select= "SELECT * from bedrooms"
    query = """
    
    CREATE VIEW bedrooms AS
    SELECT *
    FROM idealista
    WHERE (area<=100 and area>=70) 
    ORDER BY price ASC
    ;



    """
    drop_view = text(drop)
    create_view = text(query)



    with engine.connect() as connection:
        connection.execute(drop_view)
        connection.execute(create_view)

    return pd.read_sql_query(select, engine)
