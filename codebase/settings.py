import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LOCATION = os.path.join(BASE_PATH, "data")
FULL_COMMS = os.path.join(BASE_PATH, "data", 'full_comms')
MATCH_JSONS = os.path.join(BASE_PATH, "data", 'match_jsons')
HTMLS = os.path.join(BASE_PATH, "data", 'htmls')
ANALYSIS = os.path.join(BASE_PATH, "data", 'analysis_data')
SCORECARDS = os.path.join(BASE_PATH, "data", "scorecards")
CAREERS = os.path.join(BASE_PATH, "data", "careers")
paths = [
    DATA_LOCATION,
    FULL_COMMS,
    MATCH_JSONS,
    HTMLS,
    ANALYSIS,
    SCORECARDS,
    CAREERS
]

for _path in paths:
    if not os.path.exists(_path):
        os.mkdir(_path)