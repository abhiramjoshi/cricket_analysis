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
TEST_MATCH_ID = '1263466'
M = MatchData(TEST_MATCH_ID)
PLAYER_ID = '49496'

def test_aggregate_fetch(m):
    return af.get_aggregates(m, 'bat-fours')

def test_player_contributions(m):
    print('Both\n------')
    pprint(af.get_player_contribution(PLAYER_ID, m))
    print('Bat\n------')
    pprint(af.get_player_contribution(PLAYER_ID, m, 'bat'))
    print('Bowl\n------')
    pprint(af.get_player_contribution(PLAYER_ID, m, 'bowl'))

def test_cricket_totals(player_id, m):
    bat, bowl = af.cricket_totals(player_id, m)
    print('Batting\n------')
    print(bat)
    print()
    print('Bowling\n------')
    print(bowl)
    print()
if __name__ == '__main__':
    start = timeit.default_timer()
    # print(test_aggregate_fetch(M))
    # test_player_contributions(M)
    test_cricket_totals(PLAYER_ID, M)
    stop = timeit.default_timer()
    print('Time: ', abs(start-stop))
        
