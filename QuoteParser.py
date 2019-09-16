import sqlite3 as sql
import regex as re
import SearchEngine as se
import autocomplete as ac
import numpy as np

SCRIPT_NOTES_REG = "\\s*\\[.*\\]\\s*"
INTERESTING_REG  = "\\s*[\\d,]+\\s*of\\s*[\\d,]+\\s*found\\s*this\\s*interesting.*"
SENTENCE_REG = "\\.{3}|[!?\\.]"
PUNCT_REG = "[.\"-]"
STOPPING_REG = "[,.!?]"




INTERESTING_REG_PATTERN = re.compile(INTERESTING_REG)
TOTAL_ID = 7364



def quote_to_list(quote):
    """

    :param quote:
    :return:
    """
    quote = re.sub(SCRIPT_NOTES_REG,"\n",quote)
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
    sent_dict = {}
    for quote in quote_list: #going over all quotes
        sent_list = re.split(SENTENCE_REG, quote)
        for sent in sent_list:
            if (sent != "" and sent !=" "):
                sentence_list.append(sent)
                sent_dict[sent] = quote
    return sentence_list, sent_dict

def get_triplets(sentence):
    """

    :param sentence:
    :return:
    """
    triplets_list = []
    #splitting sentence to words:
    word_str = re.sub(PUNCT_REG," ",sentence)
    word_str  = re.sub("\\s\\s"," ", word_str)
    words_list = word_str.split()
    s = max(len(words_list)-3,0)
    for i in range(s): #going over all triplets
                cur_triple = " ".join(words_list[i:(i+3)])
                triplets_list.append(cur_triple)
    return triplets_list


def get_all_triplets(sent_list):
    """
    this function gets a list of sentences, and creates a list containing a tuple of all (triplet,
    sentence) pairs, where triplet is a 3-word sequence from the sentence, and sentence is the
    sentence itself.
    :param sent_list:
    :return:
    """
    all_tripts = {}
    for sent in sent_list:
        cur_tripts = get_triplets(sent)
        for triplet in cur_tripts:
            all_tripts[triplet] = (sent,find_succeeding_sequence(triplet,sent))
    return all_tripts


def find_succeeding_sequence(seq, sentence):
    """

    :param seq:
    :param sentence:
    :return:
    """
    i = sentence.find(seq)
    substring = sentence[(i+len(seq)):]
    #getting the substring up to the stopping point
    si = re.search(STOPPING_REG, substring)
    if (si):
        return seq+substring[:si.start()]
    return seq+substring


def get_best_triplets_by_searches(best_triplets, movie_name):
    """

    :param best_triplets:
    :param movie_name:
    :return:
    """
    max_searches = 0
    best_triplet = ""
    for i in range(len(best_triplets)):
        trip = best_triplets[i][0]
        query = "\""+movie_name+"\" \""+trip+"\""
        cur_searches = se.get_query_total_results(query)
        if (cur_searches > max_searches):
            max_searches = cur_searches
            best_triplet = trip
    return best_triplet,max_searches

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
    #transferring into strings

    lines_str = "\n".join(lines)
    return (lines_str, min_dist)


def best_triplets_to_lines(best_triplets, trip_dict):
    """

    :param best_triplets:
    :return:
    """
    best_lines = []
    for i in range(len(best_triplets)):
        trip = best_triplets[i][0]
        cur_line = find_succeeding_sequence(trip, trip_dict[trip])
        best_lines.append((cur_line,best_triplets[i][1]))
    return best_lines


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
    sent_list,sent_dict = get_quote_list_by_sentence(quote_list)
    #getting the best line and its editing distance
    try:
        best_lines = ac.find_best_match(sent_list,movie_name,qs)
    except:
        print("got stuck in " + str(quote_id))
        raise
    lines_str,dist = get_best_line(best_lines)
    return (quote_str,lines_str,dist)



def parse_part_autocomplete_table_row(row,quote_id,qs):
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
    sent_list,sent_dict = get_quote_list_by_sentence(quote_list)
    triplets_dict = get_all_triplets(sent_list)
    triples_list = triplets_dict.keys()
    #getting the best line and its editing distance
    try:
        best_triplets = ac.find_best_triple_match(triples_list,movie_name,triplets_dict)
    except:
        print("got stuck in " + str(quote_id))
        # write_to_partial_autocomplete_db(conn,table_name,quote_id)
        raise
    # best_lines = best_triplets_to_lines(best_triplets, triplets_dict)
    # lines_str,dist = get_best_line(best_lines)
    try:
        best_trip,max_searches = get_best_triplets_by_searches(best_triplets, movie_name)
    except:
        print("google got us stuck in " + str(quote_id))
        raise
    if (best_trip == ""):
        best_sent = ""
        best_seq = ""
    else:
        best_sent,best_seq = triplets_dict[best_trip]
    return (quote_str,best_trip,best_seq,best_sent,max_searches)







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



def write_to_autocomplete_table(conn,table_name,row,quote_str,line_str, dist):
    """

    :param conn:
    :param table_name:
    :param row:
    :param quote_str:
    :param line_Str:
    :param dist_str:
    :return:
    """

    conn.execute("INSERT INTO "+table_name+ " (ID,QUOTE,BEST_LINE,EDIT_DIST,MOVIE_NAME,MOVIE_ID,"
                "INTERESTED,TOTAL_REP) VALUES (?,?,?,?,?,?,?,?)", (row[0],quote_str,
                line_str,dist,row[2],row[3],row[4],row[5]))
    conn.commit()


def write_to_new_autocomplete_table(conn,table_name,row,quote_str,best_trip,best_seq,
                                best_sent, max_searches):
    """

    :param conn:
    :param table_name:
    :param row:
    :param quote_str:
    :param line_Str:
    :param dist_str:
    :return:
    """

    conn.execute("INSERT INTO "+table_name+ " (ID,QUOTE,BEST_TRIP,BEST_SEQ,BEST_LINE,MAX_SEARCHES,"
                                            "MOVIE_NAME,MOVIE_ID,"
                "INTERESTED,TOTAL_REP) VALUES (?,?,?,?,?,?,?,?,?,?)", (row[0],quote_str,
                best_trip,best_seq,best_sent,max_searches,row[2],row[3],row[4],row[5]))
    conn.commit()


def create_search_table():
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




def create_autocomplete_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("DB\\top_250_movie_quotes.db")
    conn.execute('''CREATE TABLE autocomplete_quotes_db
        (ID INT PRIMARY KEY     NOT NULL,
        QUOTE    TEXT NOT NULL,
        BEST_LINE TEXT NOT NULL,
        EDIT_DIST INT NOT NULL,
        MOVIE_NAME TEXT NOT NULL,
        MOVIE_ID TEXT NOT NULL,
        INTERESTED INT NOT NULL,
        TOTAL_REP INT NOT NULL);''')
    conn.commit()

def create_partial_autocomplete_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("DB\\top_250_movie_quotes.db")
    conn.execute('''CREATE TABLE partial_autocomplete_quotes_db
        (ID INT PRIMARY KEY     NOT NULL,
        QUOTE    TEXT NOT NULL,
        BEST_LINE TEXT NOT NULL,
        EDIT_DIST INT NOT NULL,
        MOVIE_NAME TEXT NOT NULL,
        MOVIE_ID TEXT NOT NULL,
        INTERESTED INT NOT NULL,
        TOTAL_REP INT NOT NULL);''')
    conn.commit()


def create_new_autocomplete_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("DB\\top_250_movie_quotes.db")
    conn.execute('''CREATE TABLE new_autocomplete_quotes_db
        (ID INT PRIMARY KEY     NOT NULL,
        QUOTE    TEXT NOT NULL,
        BEST_TRIP    TEXT NOT NULL,
        BEST_SEQ TEXT NOT NULL,
        BEST_LINE    TEXT NOT NULL,
        MAX_SEARCHES INT NOT NULL,
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
            (quote_str,line_str,dist) = parse_autocomplete_table_row(row,i,3)
            write_to_autocomplete_table(conn,table_name,row,quote_str,line_str,dist)


def write_to_partial_autocomplete_db(conn,table_name,start_id):
    for i in range(start_id,TOTAL_ID):
        table = conn.execute("SELECT * FROM top_quotes_db WHERE id="+str(i))
        for row in table:
            (quote_str,line_str,dist) = parse_part_autocomplete_table_row(row,i,3)
            # write_to_autocomplete_table(conn,table_name,row,quote_str,line_str,dist)


def write_to_new_autocomplete_db(conn,table_name,start_id):
    for i in range(start_id,TOTAL_ID):
        table = conn.execute("SELECT * FROM top_quotes_db WHERE id="+str(i))
        for row in table:
            quote_str,best_trip,best_seq,best_sent,max_searches = parse_part_autocomplete_table_row(row,i,3)
            write_to_new_autocomplete_table(conn,table_name,row,quote_str,best_trip,best_seq,
                                            best_sent,max_searches)





table_name = "new_autocomplete_quotes_db"
conn =  sql.connect("DB\\top_250_movie_quotes.db")
# create_new_autocomplete_table()
write_to_new_autocomplete_db(conn,table_name,58)
# create_partial_autocomplete_table()
# write_to_partial_autocomplete_db(conn, table_name,1312)

# table_name = "autocomplete_quotes_db"
# write_to_auto_complete_db(conn,table_name,2553)


# sent = "a very bright boy. i'm gonna make him an offer he can't refuse. to be or not to be? nothing compares to you!"
# x = find_succeding_sequence("make him an", sent)
# nir  = 1
# quote_list = quote_to_list(sent)
# sent_list = get_quote_list_by_sentence(quote_list)
# print(ac.find_best_match(sent_list,"godfather",3))