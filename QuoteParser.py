import sqlite3 as sql
import regex as re
import SearchEngine as se
import autocomplete as ac
import numpy as np

SCRIPT_NOTES_REG = "\\s*\\[.*\\]\\s*"
INTERESTING_REG  = "\\s*[\\d,]+\\s*of\\s*[\\d,]+\\s*found\\s*this\\s*interesting.*"
SENTENCE_REG = "\\.{3}|[!?\\.]"

INTERESTING_REG_PATTERN = re.compile(INTERESTING_REG)
TOTAL_ID = 7364



def quote_to_list(quote):
    """

    :param quote:
    :return:
    """
    quote = re.sub(SCRIPT_NOTES_REG,"",quote)
    quote = re.sub("\\s*\n\s*\n\s*","\n", quote)
    quote = re.sub("\\s*:\\s*\n",":",quote)
    quote = quote.split("\n")
    return quote





def process_quote_list(quote_list):
    """
    This function gets a list, in which each cell is a line in the quote, and processes it to remove end of quote notes.
    :param quote_list:
    :return:
    """
    new_quote_list = []
    quote_str = ""
    #going over all lines, to see what score they get, and whether we are done:
    for line in quote_list:
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
            new_quote_list.append(line[id:])
    return (new_quote_list,quote_str)


def get_quote_list_by_sentence(quote_list):
    """

    :param quote_list:
    :return:
    """
    sentence_list = []
    for quote in quote_list: #going over all quotes
        sent_list = re.split(SENTENCE_REG, quote)
        for sent in sent_list:
            if (sent != ""):
                sentence_list.append(sent)
    return sentence_list

def get_best_line(best_lines):
    """

    :param best_lines:
    :return:
    """
    min_dist = np.inf
    best_ind = []
    for i in range(len(best_lines)):
        cur_dist = best_lines[i][1]
        if (cur_dist < min_dist):
            min_dist = cur_dist
            best_ind = [i]
        elif (cur_dist == min_dist): #multiple choices
            best_ind.append(i)
    #getting the best lines, and their editing distance:
    lines = [best_lines[i][0] for i in best_ind]
    distances = [str(best_lines[i][1]) for i in best_ind]
    #transferring into strings

    lines_str = "\n".join(lines)
    dist_str = "\n".join(distances)
    return (lines_str, dist_str)


def parse_autocomplete_table_row(row,quote_id,qs):
    """
    :param row:
    :param quote_id:
    :param qs:
    :return:
    """
    quote = row[1]
    movie_name = row[2]
    quote_list = quote_to_list(quote)
    #getting the processed quote_list and string
    quote_list,quote_str = process_quote_list(quote_list)
    sent_list = get_quote_list_by_sentence(quote_list)
    #getting the best line and its editing distance
    best_lines = ac.find_best_match(sent_list,movie_name,qs)
    lines_str,dist_str = get_best_line(best_lines)
    return (None,None,None)






def parse_search_table_row(row, quote_id):
    """

    :param row:
    :param quote_id:
    :return:
    """
    quote = row[1]
    movie_name = row[2]
    quote = quote_to_list(quote)
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



def write_to_search_table(conn,table_name,row,quote_str,max_line, max_searches):
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


def write_to_search_db(conn,table_name,start_id):
    for i in range(start_id,TOTAL_ID):
        table = conn.execute("SELECT * FROM top_quotes_db WHERE id="+str(i))
        for row in table:
            (quote_str,max_line,max_searches) = parse_search_table_row(row,i)
            write_to_search_table(conn,table_name,row,quote_str,max_line,max_searches)


def write_to_auto_complete_db(conn,table_name,start_id):
    for i in range(start_id,TOTAL_ID):
        table = conn.execute("SELECT * FROM top_quotes_db WHERE id="+str(i))
        for row in table:
            (quote_str,max_line,max_searches) = parse_autocomplete_table_row(row,i,3)






conn =  sql.connect("DB\\top_250_movie_quotes.db")
# create_simple_table()
table_name = "parsed_quotes_db"
# write_to_search_db(conn,table_name,59)
write_to_auto_complete_db(conn,table_name,0)


# sent = "a very bright boy. i'm gonna make him an offer he can't refuse. to be or not to be? nothing compares to you!"
#
# quote_list = quote_to_list(sent)
# sent_list = get_quote_list_by_sentence(quote_list)
# print(ac.find_best_match(sent_list,"godfather",3))