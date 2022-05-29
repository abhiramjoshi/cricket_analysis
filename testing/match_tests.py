import os
import sys
import timeit
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'Base: {BASE_PATH}')
sys.path.append(BASE_PATH)

from espncricinfo.match import Match
from codebase.match_data import MatchData
from pprint import pprint

TEST_MATCH_ID = 352662
OLDEST_MATCH_ID = '62396'

def test_full_comms_grab(matchId):
    m = MatchData(match_id=matchId)
    comms = m.full_comms
    pprint({'length': len(comms)})

def print_comms_data():
    M = MatchData(match_id='489228')
    print(len(M.full_comms))
    print(type(M.full_comms))
    print(type(M.full_comms[0]))
    comms = sorted(M.get_full_comms(), key=lambda d: d['_uid'])
    print(len(comms))
    print()
    print(comms[0])
    print()
    print(comms[1000])
    print()
    print(comms[-1])

    
if __name__ == '__main__':
    start = timeit.default_timer()
    test_full_comms_grab(TEST_MATCH_ID)
    # print_comms_data()
    stop = timeit.default_timer()
    print('Time: ', start-stop)
    