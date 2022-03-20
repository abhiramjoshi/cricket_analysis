import logging
import os
from datetime import datetime

LOGDIR = './logs'

if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)

LOGFILE = os.path.join(LOGDIR, f'cricket_data_log_{datetime.now().strftime("%Y_%m_%d_%H%M")}.log')
WARNINGS_LOG = os.path.join(LOGDIR, f'cricket_data_warnings.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

f_logger = logging.FileHandler(LOGFILE)
s_logger = logging.StreamHandler()

f_logger.setLevel(logging.DEBUG)
s_logger.setLevel(logging.INFO)

f_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
s_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

f_logger.setFormatter(f_formatter)
s_logger.setFormatter(s_formatter)

logger.addHandler(f_logger)
logger.addHandler(s_logger)

class NoMatchCommentaryError(Exception):
    pass