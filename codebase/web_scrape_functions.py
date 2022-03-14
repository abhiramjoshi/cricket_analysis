from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import requests
from datetime import date
import re
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from utils import logger

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
                row.append((element.text, link))
            except TypeError:
                row.append(element.text)
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

def player_match_list(player_id, _format, match_links = False):
    url = get_statsguru_player_url(player_id, _format)
    table = read_statsguru(url, table_name='Match by match list')[0]
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
