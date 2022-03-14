from matplotlib.pyplot import vlines
import codebase.match_data as match
import codebase.settings as settings
import pandas as pd
import numpy as np
import os
from pprint import pprint
import codebase.analysis_functions as af
import utils
import codebase.web_scrape_functions as wsf
import numpy as np
from codebase.match_data import MatchData
from utils import logger
from codebase.settings import DATA_LOCATION

matchlist = wsf.get_match_list(years=['2007', ':'], finished=True)

# matchlist = [x[1] for x in matchlist]
all_comms = []

for m_id in matchlist:
    try:
        logger.info(f'Grabbing data for matchID {m_id}')
        _match = MatchData(m_id)
        comms = af.pre_transform_comms(_match)
        comm_w_labels = af.create_labels(comms, ['isWicket', 'isFour', 'isSix'], null_category='noEvent')
        all_comms.append(comm_w_labels)
    except utils.NoMatchCommentaryError:
        continue

# exit()
try:
    all_comms = pd.concat(all_comms, ignore_index=True)
    print(all_comms.size)
    print(all_comms.groupby('labels').size())
    all_comms.to_pickle(os.path.join(DATA_LOCATION, 'commentary_labels.csv'))
except ValueError:
    print('No commentary to show')