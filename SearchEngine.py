
from apiclient.discovery import build
import sqlite3 as sql
import re
import requests
from bs4 import BeautifulSoup


API_KEY = "AIzaSyD01ejGuM4fpVyV547pUshsfUfA5PI_MR8"
CUSTOM_SEARCH_ID = "018242992400709291311:pbnzbzhmref"




def pyGoogleSearch(query):
    address = 'http://www.google.com/search?q='
    newword = address + query
    page = requests.get(newword)
    soup = BeautifulSoup(page.content, 'html.parser')
    phrase_extract = soup.find(id="resultStats")
    print(phrase_extract.text)


pyGoogleSearch("huji")

def get_query_total_results(query):
    """

    :param query:
    :return:
    """
    resource = build("customsearch","v1",developerKey=API_KEY).cse()
    res = resource.list(q=query,cx = CUSTOM_SEARCH_ID).execute()
    return int(res['searchInformation']['totalResults'])



