
import requests
import json
import sqlite3 as sql
import string
import re
import numpy as np

PUNCT_REG = "[,.\"\'-]"


def get_autocomplete(query):
    URL = "http://suggestqueries.google.com/complete/search?client=firefox&q="+query
    headers = {'User-agent':'Mozilla/5.0'}
    response = requests.get(URL, headers=headers)
    result = json.loads(response.content.decode('utf-8'))
    return result[1]





def find_best_match(lines,movie_name,qs):
    """
    this function finds the best match
    :param lines: a list of lines to search
    :param movie_name: the name of the movie the lines are taken from
    :param qs: an integer, defining how many words to search from each line
    :return:
    """
    sep = " "
    best_lines = []
    min_position = 11
    for line in lines:
        if (line ==""):
            continue
        #removing punctuation and getting the required number of words:
        l = re.sub(PUNCT_REG," ", line)
        l2 = l.split()
        s = min(len(l2),qs)
        q1 = sep.join(l2[:s])
        #preparing the query and getting the autocomplete results.
        query = movie_name+" "+ q1
        auto_comp = get_autocomplete(query)
        #finding the best match from the autocomplete matches:
        min_edit_dist = np.inf
        best_ind = -1
        for i in range(len(auto_comp)):
            sug = auto_comp[i]
            sug = re.sub(PUNCT_REG, " ", sug)
            cur_dist = levenshteinDistance(l,sug)
            if (cur_dist < min_edit_dist):
                best_ind = i
                min_edit_dist = cur_dist
        if (best_ind == -1):
            continue
        cur_best_suggest = auto_comp[best_ind]
        #checking if the current best match has better position:
        if (min_position > best_ind):
            min_position = best_ind
            best_lines = [(line,min_edit_dist)]
        elif (min_position == best_ind):
            best_lines.append((cur_best_suggest, min_edit_dist))
    return best_lines






def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# movie_name = "godfather"
# lines = ["i'm gonna make him an offer", "oh danny boy the winds are calling", "ain't no sunshine when she's gone"]
# find_best_match(lines, movie_name, 2)


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

