import requests
from bs4 import BeautifulSoup
import re
import sqlite3 as sql


URL_START = "https://www.imdb.com/title/"
URL_END = "/quotes/?tab=qt&ref_=tt_trv_qu"
INTERESTING_LINE_REG = "\\s*(\\d+)\\s*\\w*\\s*(\\d+)\\s*(found this interesting)"
TOP_250_URL = "https://www.imdb.com/chart/top"
MOVIE_TITLE_REGEX = "\\n\\s*\\d\\.\\n\\s*((\\w+\\s*)+)\\n.*"
DB_PATH = "DB\\top_250_movie_quotes.db"
TABLE_NAME = "top_quotes_db"

# init_url = "https://www.imdb.com/title/tt0068646/quotes/?tab=qt&ref_=tt_trv_qu"
# url = requests.get(init_url)
# t = url.text
# soup =  BeautifulSoup(t)
# nir = soup.findAll("div", {"class": "quote soda sodavote odd"})
# a = nir[0].text.split("\n")
# f = open("trial.txt", mode="w")
# print(nir[0].txet)
# f.write(nir[0].text)
# f.close()



# f = open("trial.txt","r")
# lines = f.readlines()
# str_start = ""
# for line in lines:
#     # line = line.replace("\\'","'")
#     line = line.replace("\n","")
#     m = re.match(INTERESTING_LINE_REG,line)
#     if (m):
#         print(line)
#         print(m.groups()[0])



def parse_top_250(db_path, table_name):
    """

    :param db_path:
    :param table_name:
    :return:
    """
    #opening url:
    quote_id = 0
    url = requests.get(TOP_250_URL)
    soup = BeautifulSoup(url.text)
    title_div = soup.findAll("td",{"class": "titleColumn"})
    for title in title_div:
        movie_name = title.contents[1].text
        movie_id = title.contents[1].attrs["href"][7:-1]
        quote_id = parse_url(movie_id,movie_name, db_path,table_name,quote_id)



def parse_url(movieID,movie_name, db_path,table_name, quote_init_id):
    """

    :param movieID:
    :param db_name:
    :return:
    """
    quote_id = quote_init_id
    conn = sql.connect(db_path)
    init_url = URL_START+movieID+URL_END
    url = requests.get(init_url)
    t = url.text
    soup =  BeautifulSoup(t)
    quote_div = soup.findAll("div", {"class": "quote soda sodavote odd"})
    for q in quote_div:
        tup = add_quotes(q.text)
        #updating the table:
        conn.execute("INSERT INTO "+table_name+" (ID,QUOTE,MOVIE_NAME,MOVIE_ID,INTERESTED,"
                                               "TOTAL_REP) VALUES "
                                               "(?,?,?,?,?,?)",(quote_id,tup[0],movie_name,
                                                                movieID,tup[1],
                                                              tup[2]))
        quote_id += 1
    conn.commit()
    return quote_id



def add_quotes(quote_text):
    """

    :param quote_text:
    :return:
    """
    #splitting the quote text:
    quote_lines = quote_text.split("\n")
    quote_str = ""
    n_interested =0
    n_total = 0
    for line in quote_lines:
        m = re.match(INTERESTING_LINE_REG, line)
        if (m): #found match, getting the interesting groups:
            g = m.groups()
            n_interested = g[0]
            n_total = g[1]
            break
        quote_str += line + "\n"
    return (quote_str, n_interested, n_total)


#----------------------------- SQL helping functions ------------------------------------

def create_simple_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("DB\\top_250_movie_quotes.db")
    conn.execute('''CREATE TABLE top_question_quotes_db
        (ID INT PRIMARY KEY     NOT NULL,
        QUOTE    TEXT NOT NULL,
        MOVIE_NAME TEXT NOT NULL,
        MOVIE_ID TEXT NOT NULL,
        INTERESTED INT NOT NULL,
        TOTAL_REP INT NOT NULL);''')
    conn.commit()

create_simple_table()
