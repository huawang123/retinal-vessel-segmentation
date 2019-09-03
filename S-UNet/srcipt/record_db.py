import sqlite3
from datetime import datetime

DB_NAME = ".history.db"


CREATE_TABLE_SQL = """
    create table if not exists exprs (
        id integer primary key,
        name text not null,
        config_name text not null,
        config_content text not null,
        driver text not null,
        time datetime not null
    );
"""


def _init_database():
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(CREATE_TABLE_SQL)
    finally:
        conn.close()


def start_expr(expr_name: str, driver_name: str, config_name: str, config_content: str):
    conn = sqlite3.connect(DB_NAME)
    _init_database()
    c = conn.cursor()
    c.execute("INSERT INTO exprs (name, config_name, config_content, driver, time) VALUES (?, ?, ?, ?, ?)", (
        expr_name,
        config_name,
        config_content,
        driver_name,
        datetime.now()
    ))
    expr_id = c.lastrowid
    conn.commit()
    conn.close()
    return expr_id


TEMPLATE = """
experiment {} start !!!
expr_name: {}
driver_name: {}
config_name: {}
config_content:\n{}\n
"""


def text_report(expr_id: str, expr_name: str, driver_name: str, config_name: str, config_content: str):
    return TEMPLATE.format(expr_id, expr_name, driver_name, config_name, config_content)

