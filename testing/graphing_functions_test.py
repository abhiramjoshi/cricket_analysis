import os
import sys
import timeit
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'Base: {BASE_PATH}')
sys.path.append(BASE_PATH)
from espncricinfo.match import Match
from codebase.match_data import MatchData
from pprint import pprint
import codebase.analysis_functions as af
import codebase.web_scrape_functions as wsf
import codebase.graphing_functions as gf
TEST_MATCH_ID = '343730'
M = MatchData(TEST_MATCH_ID)
PLAYER_IDS = ['253802', '303669', '277906', '267192']


def multi_player_graphing():
    gf.graph_multi_player_batting_careers()

if __name__ == '__main__':
    start = timeit.default_timer()
    gf.graph_multi_player_batting_careers(PLAYER_IDS, dates="2021-02-21:")
    stop = timeit.default_timer()
    print('Time: ', abs(start-stop))