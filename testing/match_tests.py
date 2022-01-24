import os
import sys
import timeit
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'Base: {BASE_PATH}')
sys.path.append(BASE_PATH)

from espncricinfo.match import Match
from codebase.match_data import MatchData
import pprint

TEST_MATCH_ID = '1263466'

def test_full_comms_grab(matchId):
    m = MatchData(match_id=matchId)
    comms = m.full_comms
    pprint.pprint({'length': len(comms)})

if __name__ == '__main__':
    start = timeit.default_timer()
    test_full_comms_grab(TEST_MATCH_ID)
    stop = timeit.default_timer()
    print('Time: ', start-stop)
    
