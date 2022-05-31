import os
import sys
import timeit
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'Base: {BASE_PATH}')
sys.path.append(BASE_PATH)

from espncricinfo.match import Match
import codebase.web_scrape_functions as wsf
import pprint


def test_tables(url, table_name='Match by match list'):
    result = wsf.read_statsguru(url, table_name)
    return result

def test_player_match_list(player_id, _format = 'test', dates=None):
    result = wsf.player_match_list(player_id, dates, _format)
    return result

def test_get_match_list():
    wsf.get_match_list()

if __name__ == '__main__':
    URL = 'https://stats.espncricinfo.com/ci/engine/player/253802.html?class=1;orderby=start;template=results;type=allround;view=match;wrappertype=text'
    TEST_MATCHES_URL = 'https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=1;id=2022;type=year;wrappertype=text'
    PLAYER_ID = '253802'
    start = timeit.default_timer()
    # tables = test_tables(TEST_MATCHES_URL, table_name=None)
    match_list = test_player_match_list(PLAYER_ID)
    # match_list = wsf.get_match_list(years=['2023'])
    stop = timeit.default_timer()
    print('Time taken: ', stop-start)
    print(match_list)
    print(len(match_list))
    # pprint.pprint(tables)
    # print(type(tables[0]))