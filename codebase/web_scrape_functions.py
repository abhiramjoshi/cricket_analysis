from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import requests
from datetime import date
import re
from collections import defaultdict
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from utils import logger
import os
from codebase.settings import SCORECARDS
import utils

BASE_STATS_URL = "https://stats.espncricinfo.com"

FORMATS = {
        'test':1,
        'odi':2,
        't20i':3,
        
    }

def create_retry_session(total = None, connect = 3, backoff_factor = 0.5):
    logger.debug('Creating session to manage retries and backoff time')
    session = requests.Session()
    retry = Retry(total = total, connect=connect, backoff_factor=backoff_factor)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_statsguru_player_url(player_id, _format):
    return f'https://stats.espncricinfo.com/ci/engine/player/{player_id}.html?class={FORMATS[_format]};orderby=start;template=results;type=allround;view=match;wrappertype=text'

def get_statsguru_matches_url(year, _format):
    return f"https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class={FORMATS[_format]};id={year};type=year"

def read_table(table_html:BeautifulSoup):
    headers_html = table_html.find('thead').find('tr')
    headers = [header.text if header.text != '' else str(i) for i,header in enumerate(headers_html.find_all('th'))]
    body_html = table_html.find('tbody')
    table_body = []
    for row_html in body_html.find_all('tr'):
        row = []
        for element in row_html.find_all('td'):
            try:
                link = element.find('a')['href']
                row.append((element.text.replace('\xa0', ''), link))
            except TypeError:
                row.append(element.text.replace('\xa0', ''))
        table_body.append(row)
    # table_body.insert(0, headers)
    return headers, table_body

def read_statsguru_table(table_html:BeautifulSoup):
    headers_html = table_html.find('thead').find('tr')
    headers = [header.text if header.text != '' else str(i) for i,header in enumerate(headers_html.find_all('th'))]
    body_html = table_html.find('tbody')
    table_body = []
    for row_html in body_html.find_all('tr'):
        row = []
        for element in row_html.find_all('td'):
            try:
                link = element.find('a')['href']
                row.append((element.text.replace('\xa0', ''), link))
            except TypeError:
                row.append(element.text.replace('\xa0', ''))
        table_body.append(row)
    # table_body.insert(0, headers)
    return headers, table_body

def read_statsguru(url, table_name = None):
    session = create_retry_session()
    logger.debug(f'Sending get request to {url}')
    response = session.get(url)

    html = BeautifulSoup(response.content, 'html.parser')
    logger.debug(f'Processing table html, requested table name is {table_name}')
    if table_name:
        try:
            table_html = html.find('caption', text=table_name).parent
            table_html = table_html.wrap(html.new_tag('div'))
        except AttributeError:
            logger.error('Table does not exist')
            return None
    else:
        table_html = html
    tables_html = table_html.find_all('table')
    tables = []
    for table in tables_html:
        logger.debug('Attempting to load table from given html')
        try:
            h, tb = read_statsguru_table(table)
            tables.append(pd.DataFrame(tb, columns=h))
        except AttributeError:
            continue
    #tables = pd.read_html(table_html, flavor='bs4')
    for table in tables:
        #remove '-' and add na
        logger.debug('Replacing null values in table')
        table.replace('-', np.nan, inplace=True)
        table.replace('', np.nan, inplace=True)
        #remove null columns
        table.dropna(axis=1, how='all', inplace=True)
        
    return tables

def match_ids_and_links(table, match_links):
    match_list = list(map(lambda x: BASE_STATS_URL + x[1], table.iloc[:,-1]))
    match_id_func = lambda x: re.match('\S+/engine/match/(\d+).html', x).group(1)
    match_ids = list(map(match_id_func, match_list))
    if match_links:
        return list(zip(match_ids, match_list))
    return match_ids

def get_player_match_list(player_id, dates=None, _format='test', match_links = False):
    """
    Get matches for a player

    Returns: IDs of matches

    player_id: Cricinfo object id for player
    dates: The dates between which the careers should be graphed, date format YYYY-MM-DD:YYYY-MM-DD
    _format: 'test'| 'odi' | 't20'
    match_links: Return the link to the cricinfo match page as well as the ids
    """
    logger.info("Getting match list for player: %s Dates: %s", player_id, dates)
    url = get_statsguru_player_url(player_id, _format)
    table = read_statsguru(url, table_name='Match by match list')[0]
    table['Start Date'] = pd.to_datetime(table['Start Date'], format='%d %b %Y')
    if dates:
        dates = dates.split(':')
        if dates[0]:
            table = table[table['Start Date'] >= dates[0]]
        if dates[1]:
            table = table[table['Start Date'] < dates[1]]
    matches = match_ids_and_links(table, match_links)
    return matches

def get_match_list(years=[date.today().year], _format='test', match_links=False, finished=False):
    """Returns list of match IDs for a given years, or all records if all is selected"""
    if not isinstance(years, list):
        if isinstance(years, str):
            years = [years]
        else:
            years = list(years)

    if years[0] == 'All':
        years = [str(x+1) for x in range(1877, date.today().year)]
    elif len(years) == 2:
        if years[0] == ':':
            years = [str(x) for x in range(1877, int(years[1]) + 1)]
        elif years[1] == ':':
            years = [str(x) for x in range(int(years[0]), date.today().year + 1)]
        else:
            years = [str(x) for x in range(int(years[0]), int(years[1]) + 1)]
    
    logger.info(f'Fetching match lists from {years[0]}-{years[-1]}')    
    
    matches = []
    for year in years:
        logger.info(f'Getting match list for {year}')
        if int(year) > date.today().year:
            logger.warning(f'Year {year} is in the future, match data retrieval operation for this year will be skipped')
            continue
        url = get_statsguru_matches_url(year, _format)
        try:
            table = read_statsguru(url, table_name='Match results')[0]
        except IndexError:
            continue
        if finished:
            table = table.dropna(subset=['Winner'])
        matches += match_ids_and_links(table, match_links)
        logger.info(f'Collected match ids for {year}')
    
    return matches

def player_id_from_link(value:tuple[str], playername=True):
    logger.debug('Player_ID_from_link value: %s', value)
    player_name = value[0].replace('(c)', '').replace('†', '').strip().lower().replace(' ', '-').replace("'", '-').replace(".", '-') #create player name by cricinfo's convention
    player_name = player_name.replace('--', '-').strip('-') #replace double dashes and dashes at the ends of the strings
    if playername:
        pattern = '/player/[a-z\-]+-(\d+)' #catch all pattern
        #pattern = f'/player/{player_name}-(\d+)'
    else:
        pattern = f'/player/(\d+)'
    matches = re.search(pattern, value[1])
    logger.debug('Id matches object: %s', matches)
    player_object_id = matches.group(1)
    return (value[0], player_object_id)

def get_match_scorecard(url, match_id, try_local=True, save=True, skip_ret_hurt_err=True, retries=3):
    if try_local:
        if os.path.exists(os.path.join(SCORECARDS, f'{match_id}_scorecard.json')):
            data = utils.load_data(match_id, suffix='scorecard', data_folder=SCORECARDS)
            if data:
                return data
    for i in range(retries):
        try:
            session = create_retry_session()
            logger.debug(f'Sending get request to {url}. Retry: {i+1}')
            response = requests.get(url)
            #logger.debug([response.url for response in response.history])
            sleep(i)
            redirect_url = response.url
            logger.debug("Redirected match URL: %s", redirect_url)
            if 'live-cricket-score' in redirect_url:
                redirect_url = redirect_url.replace('/live-cricket-score', '')
            if 'full-scorecard' not in redirect_url:
                redirect_url = redirect_url+'/full-scorecard'
            logger.debug('Redirected URL after handling is %s, sending request here instead', redirect_url)
            response = session.get(redirect_url)
            html = BeautifulSoup(response.content, 'html.parser')
            scorecard_block = html.find('div', {'class':"lg:ds-container lg:ds-mx-auto lg:ds-px-5 lg:ds-pt-4"}).findChild("div", {'class':'ds-grow'})
            title_card = scorecard_block.find('div',{"class":"ds-w-full ds-bg-fill-content-prime ds-overflow-hidden ds-rounded-xl ds-border ds-border-line"})
            scorecards = title_card.find_next_sibling('div', {'class':'ds-mt-3'})
            scorecard = scorecards.findAll('table')
            break
        except AttributeError:
            continue
    logger.debug('Fetched scorecard')
    logger.debug('Parsing scorecard')
    tables = []
    for _section in scorecard:
        try:
            tables.append(read_table(_section))
        except AttributeError as e:
            logger.debug(e)
            continue

    def how_out(value, skip_ret_hurt_err=True):
        if 'st ' in value:
            return 'stumped'
        if 'c ' in value:
            return 'caught'
        if 'run out' in value:
            return 'run_out'
        if 'lbw ' in value:
            return 'lbw'
        if 'not out' in value:
            return False
        if 'retired hurt' in value:
            if not skip_ret_hurt_err:
                raise utils.RetiredHurtError
            return False
        if ' b ' in value or 'b ' in value:
            return 'bowled'
        else:
            return True

    def total(row):
        _totals = {}
        total_types = {
            'overs': '(\d{1,3}.{0,1}\d{0,1}) Ov',
            'run_rate':'RR: (\d{1,3}.\d{2})',
            'minutes':'(\d{1,4}) Mts',
        }
        for t in total_types:
            try:
                _totals[t] = re.search(total_types[t], row[1]).group(1)
            except AttributeError:
                logger.debug('failed to match: %s, Pattern: %s', row[1], total_types[t])
                continue
        
        _totals['score'] = row[2]

        return _totals

    def extras(row):
        _extras = {}
        extra_type = {
            'byes':'b (\d{1,3})',
            'leg_byes':'lb (\d{1,3})',
            'wides': 'w (\d{1,3})',
            'no_balls': 'nb (\d{1,3})'
        }
        for e in extra_type:
            try:
                _extras[e] = re.search(extra_type[e], row[1]).group(1)
            except AttributeError:
                logger.debug('failed to match: %s, Pattern: %s', row[1], extra_type[e])
                continue
        return _extras

    match_scorecard = defaultdict(dict)
    for i,table in enumerate(tables):
        inning = (i//2) + 1
        headers = table[0]
        stats = []

        if headers[0] == 'BATTING':
            section = 'batting'
            headers[0] = 'batsman'
        else:
            section = 'bowling'
            headers[0] = 'bowler'

        for row in table[1]:
            if len(row) != len(headers):
                if row[0] == 'Extras':
                    match_scorecard[f'inning_{inning}']['extras'] = extras(row) 
                if row[0] == 'TOTAL':
                    match_scorecard[f'inning_{inning}']['totals'] = total(row)
                continue
            row_dict = {(header if header != '\xa0' else 'out'):(row[i] if i != 1 else how_out(row[i], skip_ret_hurt_err=skip_ret_hurt_err)) for i,header in enumerate(headers)}
            row_dict[list(row_dict.keys())[0]] = player_id_from_link(row_dict[list(row_dict.keys())[0]])
            stats.append(row_dict)

        match_scorecard[f'inning_{inning}'][section] = stats

        if save:
            utils.save_data(match_id=match_id, data=dict(match_scorecard), suffix='scorecard', data_folder=SCORECARDS, serialize=False)
    return dict(match_scorecard)

def get_player_json(player_id):
    session  = create_retry_session()
    response = session.get(f"http://core.espnuk.org/v2/sports/cricket/athletes/{player_id}")
    if response.status_code == 404:
        raise utils.PlayerNotFoundError
    else:
        return response.json()