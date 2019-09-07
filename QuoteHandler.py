import sqlite3 as sql



class Quote:
    """
    this class defines a quote type
    """
    def __init__(self, text,movie_name,movie_id,quote_id,interested,total_replies):
        self.text = text
        self.movie_name = movie_name
        self.movie_id = movie_id
        self.quote_id = quote_id
        self.interested = interested
        self.total_replies = total_replies




DB_PATH = "DB\\top_250_movie_quotes.db"
READ_TABLE_NAME = "top_quotes_db"
WRITE_TABLE_NAME = "top_question_quotes_db"



def parse_row_to_quote(row):
    """

    :param row:
    :return:
    """
    #getting the quote attributes from the table row:
    quote_id = int(row[0])
    quote_text = row[1]
    movie_name = row[2]
    movie_id = row[3]
    interested = int(row[4])
    total_replies = int(row[5])
    quote = Quote(quote_text,movie_name,movie_id,quote_id, interested,total_replies)
    return quote


def get_quotes_from_table(db_path, table_name):
    """

    :param db_path:
    :param table_name:
    :return:
    """
    quote_lst = []
    conn =  sql.connect(db_path)
    table = conn.execute("SELECT * FROM "+table_name)
    for row in table: #going over all rows
        #checking if the quote fits our purpose:
        if (check_question_mark(row[1])):
            quote_lst.append(parse_row_to_quote(row))
    return quote_lst


def check_question_mark(quote_text):
    """

    :param quote_text:
    :return:
    """
    return ("?" in quote_text)


def write_top_quotes(quotes_lst,db_path, table_name, n):
    """

    :param quotes_lst:
    :param db_path:
    :param table_name:
    :param n:
    :return:
    """
    conn = sql.connect(db_path)
    #sorting the quotes list:
    quotes_lst.sort(key = get_interested,reverse = True)
    for i in range(n):
        quote = quotes_lst[i]
        #writing quote to the DB
        conn.execute("INSERT INTO "+table_name+" (ID,QUOTE,MOVIE_NAME,MOVIE_ID,INTERESTED,"
                    "TOTAL_REP) VALUES " "(?,?,?,?,?,?)",(quote.quote_id,quote.text,
                    quote.movie_name, quote.movie_id,quote.interested,quote.total_replies))
    conn.commit()






def get_interested(quote):
    """

    :param quote:
    :return:
    """
    return quote.interested


def quote_handler(n):
    """

    :param n:
    :return:
    """
    #getting the quotes
    quotes_lst = get_quotes_from_table(DB_PATH,READ_TABLE_NAME)
    write_top_quotes(quotes_lst, DB_PATH, WRITE_TABLE_NAME,n)


quote_handler(200)