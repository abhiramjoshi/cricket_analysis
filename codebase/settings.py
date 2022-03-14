import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LOCATION = os.path.join(BASE_PATH, "data")
MATCH_JSONS = os.path.join(BASE_PATH, "data", 'match_jsons')
HTMLS = os.path.join(BASE_PATH, "data", 'htmls')

for _path in [DATA_LOCATION, MATCH_JSONS, HTMLS]:
    if not os.path.exists(_path):
        os.mkdir(_path)