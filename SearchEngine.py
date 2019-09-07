
from apiclient.discovery import build

api_key = "AIzaSyD01ejGuM4fpVyV547pUshsfUfA5PI_MR8"
custom_search_id = "018242992400709291311:pbnzbzhmref"

print("\"shutter island\" \"to live as a monster\"")
resource = build("customsearch","v1",developerKey=api_key).cse()
res = resource.list(q="\"shutter island\" \"to live as a monster\"",
                    cx = custom_search_id).execute()
print(res['searchInformation']['totalResults'])