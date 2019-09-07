
import requests
import json
import sqlite3 as sql

def get_autocomplete(query):
    URL = "http://suggestqueries.google.com/complete/search?client=firefox&q="+query
    headers = {'User-agent':'Mozilla/5.0'}
    response = requests.get(URL, headers=headers)
    result = json.loads(response.content.decode('utf-8'))
    return result[1]



def write_query_to_table(conn,table_name,query):
    """

    :param conn: sqlite3 connection object
    :param table_name: the name of the table to insert to
    :param query: the query to get the autocomplete for
    :return:
    """
    #getting the autocomplete suggestions
    autocomp = get_autocomplete(query)
    #inserting to the table:
    conn.execute("INSERT INTO "+table_name+ " (QUERY,AUTO1,AUTO2,AUTO3,AUTO4,AUTO5,AUTO6,AUTO7,"
                "AUTO8,AUTO9,AUTO10) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (query,autocomp[0],
                                                                       autocomp[1],
                 autocomp[2],autocomp[3],autocomp[4],autocomp[5],autocomp[6],autocomp[7],
                 autocomp[8],autocomp[9]))
    conn.commit()


def update_autocomp_table(table_name,queries):
    conn = sql.connect("DB\\autocomplete.db")
    for query in queries:
        write_query_to_table(conn,table_name,query)



def create_simple_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("DB\\autocomplete.db")
    conn.execute('''CREATE TABLE test_db
        (QUERY TEXT PRIMARY KEY     NOT NULL,
        AUTO1    TEXT,
        AUTO2    TEXT,
        AUTO3    TEXT,
        AUTO4    TEXT,
        AUTO5    TEXT,
        AUTO6    TEXT,
        AUTO7    TEXT,
        AUTO8    TEXT,
        AUTO9    TEXT,
        AUTO10    TEXT);''')
    conn.commit()

queries = ["Bibi","hebrew","quent"]
update_autocomp_table("test_db",queries)