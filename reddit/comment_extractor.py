"""
this file is used to extract comments off a reddit subreddit, into a new database.
"""

import bz2
import json
import sys
import os.path
import praw
import sqlite3
import CommentNode as CN
import datetime






def scrub(table_name):
    """
    this function is used to scrub off unwanted chars in the table name
    :param table_name: the name of the table to scrub.
    :return:
    """
    return ''.join( chr for chr in table_name if chr.isalnum() )


def get_date(comment):
    """

    :param comment:
    :return:
    """
    date = comment.created
    time = datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')
    return time


def create_db(submission, conn, name, washup = False):
    """

    :param submission: the reddit submission to mine.
    :param conn: an sqlite connector
    :param name: the name of the table to create
    :return:
    """

    conn.execute('''CREATE TABLE '''+ name+'''
    (ID TEXT PRIMARY KEY     NOT NULL,
    BODY    TEXT NOT NULL,
    PARENT_ID TEXT NOT NULL,
    DATE TEXT NOT NULL,
    SCORE INT NOT NULL);''')
    print("table " + name+" created successfully")
    date = get_date(submission)
    conn.execute("INSERT INTO "+name+" (ID,BODY,PARENT_ID,DATE, SCORE) VALUES (?,?, ?,?,?)", ("0",
                                                                              submission.title,
                                                                              str(submission.id),
                                                                              date, submission.score))
    #submitting comments to the table
    conn.commit()
    submission.comments.replace_more(None)
    i = 0
    for comment in submission.comments.list():
        if ( i%250 == 0):
            print(i)
        i+=1
        comment_id_str = "'"+str(comment.id)+"'"
        comment_body_str = "'" + str(comment.body) + "'"
        comment_parent_id = "'"+str(comment.parent_id)+"'"
        score = comment.score
        date = "'"+get_date(comment)+"'"
        conn.execute(("INSERT INTO " + name+" (ID,BODY,PARENT_ID,DATE,SCORE) VALUES (?,?,?,?,?)"),
                     (comment_id_str,comment_body_str,comment_parent_id, date,score))

    conn.commit()


def washup(subreddit_name, upto):
    """

    :param subreddit_name:
    :param upto:
    :return:
    """
    conn = sqlite3.connect("reddit_new_"+subreddit_name+".db")
    for i in range(1,upto+1):
        #dropping table
        conn.execute("DROP TABLE " + subreddit_name+str(i))
    conn.close()

def create_subreddit_table(subreddit_name, reddit):
    """
    this function creates a full databse, consisting of tables. Each table is defined by a single
    reddit post.
    :param subreddit_name: the name of the subreddit to mine
    :param reddit: the reddit praw object.
    :return:
    """
    #creating the database to store the trees:
    conn = sqlite3.connect("reddit_new_"+subreddit_name+".db")
    print("reddit_"+subreddit_name+".db")
    table_names = []
    subreddit = reddit.subreddit(subreddit_name).hot(limit =1000)
    i = 1
    #going through all submissions
    for submission in subreddit:
        name = subreddit_name+str(i)
        table_names.append(name)
        i+=1
        create_db(submission, conn,name)
    conn.close()
    return table_names


