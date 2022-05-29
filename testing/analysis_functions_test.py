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
TEST_MATCH_ID = '1278683'
M = MatchData(TEST_MATCH_ID)
PLAYER_ID = '253802'

def test_aggregate_fetch(m):
    return af.get_aggregates(m, 'bat-fours')

def test_player_contributions(m, is_object_id=False, by_innings=False):
    print('Both\n------')
    pprint(af.get_player_contribution(PLAYER_ID, m, is_object_id=is_object_id, by_innings=by_innings))
    print('Bat\n------')
    pprint(af.get_player_contribution(PLAYER_ID, m, 'bat', is_object_id=is_object_id, by_innings=by_innings))
    print('Bowl\n------')
    pprint(af.get_player_contribution(PLAYER_ID, m, 'bowl', is_object_id=is_object_id, by_innings=by_innings))

def test_cricket_totals(player_id, m, is_object_id=False, by_innings=False):
    bat, bowl = af.cricket_totals(player_id, m, is_object_id=is_object_id, by_innings=by_innings)
    print('Batting\n------')
    print(bat)
    print()
    print('Bowling\n------')
    print(bowl)
    print()

def test_get_career_batting_graph(player_id, _format='test', window_size=12):
    af.get_career_batting_graph(player_id, _format=_format, window_size=window_size)

def test_running_average(player_id, _format='test'):
    running_average = af.get_running_average(player_id)

def test_get_figures_from_scorecard(player_id, match, _type):
    figures = af.get_figures_from_scorecard(player_id, match, _type)
    return figures

if __name__ == '__main__':
    start = timeit.default_timer()
    # print(test_aggregate_fetch(M))
    # test_player_contributions(M, is_object_id=True, by_innings=True)
    # test_cricket_totals(PLAYER_ID, M, is_object_id=True, by_innings=True)
    # test_running_average(PLAYER_ID)
    # test_get_career_batting_graph(PLAYER_ID)
    print(test_get_figures_from_scorecard(PLAYER_ID, M, 'bat'))
    stop = timeit.default_timer()
    print('Time: ', abs(start-stop))
        
