import sqlite3 as sql
import regex as re
import SearchEngine as se

SCRIPT_NOTES_REG = "\\s*\\[.*\\]\\s*"
INTERESTING_REG  = "\\s*[\\d,]+\\s*of\\s*[\\d,]+\\s*found\\s*this\\s*interesting.*"

INTERESTING_REG_PATTERN = re.compile(INTERESTING_REG)
TOTAL_ID = 7364



def parse_table_row(row, quote_id):
    """

    :param row:
    :param quote_id:
    :return:
    """
    quote = row[1]
    movie_name = row[2]
    quote = re.sub(SCRIPT_NOTES_REG,"",quote)
    quote = re.sub("\\s*\n\s*\n\s*","\n", quote)
    quote = re.sub("\\s*:\\s*\n",":",quote)
    quote = quote.split("\n")
    quote_str = ""
    max_searches = 0
    max_line = ""
    #going over all lines, to see what score they get, and whether we are done:
    for line in quote:
        if (line == ""):

            continue
        #checking if we reached the end:
        m = INTERESTING_REG_PATTERN.match(line)
        if (m): #reached the end of the quote
            break
        else:
            quote_str += line+"\n"
            #removing the character name from the quote
            id = line.find(":") + 1
            query = "\""+movie_name+"\" \"" +line[id:]+"\""
            try:
                cur_searches = se.get_query_total_results(query)
            except:
                print("the id we got stuck in is " + str(quote_id))
                raise
            if (cur_searches  > max_searches):
                max_searches = cur_searches
                max_line = line

    return (quote_str,max_line, max_searches)



def write_to_table(conn,table_name,row,quote_str,max_line, max_searches):
    """

    :param conn:
    :param table_name:
    :param row:
    :param quote_str:
    :param max_line:
    :param max_searches:
    :return:
    """
    conn.execute("INSERT INTO "+table_name+ " (ID,QUOTE,BEST_LINE,SEARCHES,MOVIE_NAME,MOVIE_ID,"
                "INTERESTED,TOTAL_REP) VALUES (?,?,?,?,?,?,?,?)", (row[0],quote_str,
                max_line,max_searches,row[2],row[3],row[4],row[5]))
    conn.commit()




def create_simple_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("DB\\top_250_movie_quotes.db")
    conn.execute('''CREATE TABLE parsed_quotes_db
        (ID INT PRIMARY KEY     NOT NULL,
        QUOTE    TEXT NOT NULL,
        BEST_LINE TEXT NOT NULL,
        SEARCHES INT NOT NULL,
        MOVIE_NAME TEXT NOT NULL,
        MOVIE_ID TEXT NOT NULL,
        INTERESTED INT NOT NULL,
        TOTAL_REP INT NOT NULL);''')
    conn.commit()


def write_to_new(conn,table_name,start_id):
    for i in range(start_id,TOTAL_ID):
        table = conn.execute("SELECT * FROM top_quotes_db WHERE id="+str(i))
        for row in table:
            (quote_str,max_line,max_searches) = parse_table_row(row,i)
            write_to_table(conn,table_name,row,quote_str,max_line,max_searches)




conn =  sql.connect("DB\\top_250_movie_quotes.db")
# create_simple_table()
table_name = "parsed_quotes_db"
write_to_new(conn,table_name,59)


