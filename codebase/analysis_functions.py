from collections import defaultdict
import codebase.match_data as match
import codebase.web_scrape_functions as wsf
import numpy as np
import pandas as pd
import sklearn.utils
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import utils
from utils import logger
from collections.abc import Iterable
from datetime import datetime
import espncricinfo.exceptions as cricketerrors

def pre_transform_comms(match_object:match.MatchData):
    logger.info(f'Pre-transforming match commentary for {match_object.match_id}')
    df = pd.DataFrame.from_dict(match_object.get_full_comms())
    if df.empty:
        raise utils.NoMatchCommentaryError
    logger.debug(f"Columns of commentary from match {match_object.match_id}")
    logger.debug(f'Columns: {df.columns}\nSize: {df.size}')
    map_players(match_object, df)
    logger.info(f'{match_object.match_id}: Processing text commentary fields')
    process_text_comms(df)
    logger.info(f'{match_object.match_id}: Processing bowler runs')
    df['bowlerRuns'] = df['batsmanRuns'] + df['wides'] + df['noballs']
    innings_map = {inning['innings_number']: inning['team_id'] for inning in match_object.innings_list}
    df['battingTeam'] = df['inningNumber'].map(innings_map)
    return df

def dates_from_age(player_id, age_range):
    if age_range:
        player_dob = get_player_dob(player_id)
        try:
            player_age = age_range.split(':')
            younger_bound = player_age[0]
            if not bool(younger_bound):
                younger_bound = player_dob
            else:
                younger_bound = int(younger_bound)
                younger_bound = player_dob.replace(year=player_dob.year+int(younger_bound)).strftime('%Y-%m-%d')
            try:
                older_bound = player_age[1]
                if not bool(older_bound):
                    older_bound = datetime.now()
                else:
                    older_bound = player_dob.replace(year=player_dob.year+int(older_bound)).strftime('%Y-%m-%d')
            except IndexError:
                older_bound = int(younger_bound)+1
        except ValueError:
            logger.error('Invalid value for player age')

        return f"{younger_bound}:{older_bound}"
    return None

def get_balls_event(comms:pd.DataFrame, column_name:str, value, negative=False):
    if negative:
        event_df = comms[comms[column_name] != value]    
    else:
        event_df = comms[comms[column_name] == value]
    return event_df

def get_player_dob(player_id):
    json = wsf.get_player_json(player_id)
    age = datetime.strptime(json['dateOfBirth'], '%Y-%m-%dT%H:%MZ')
    return age

def get_player_map(match_object:match.MatchData, map_column:str='card_long', map_id='player_id'):
    return {int(player[map_id]):player[map_column] for player in match_object.all_players}

def map_players(match_object:match.MatchData, comms:pd.DataFrame):
    logger.debug(f'{match_object.match_id}: Mapping player names')
    player_map = get_player_map(match_object)
    comms["batsmanName"] = comms["batsmanPlayerId"].map(player_map)
    comms["bowlerName"] = comms["bowlerPlayerId"].map(player_map)

def series_to_df(series:pd.Series, column_names:list, remove_index:bool=True):
    if isinstance(series, pd.Series):
        df = series.to_frame()
    else:
        df = series
    df.rename(columns={df.columns[0]: column_names[1]}, inplace=True)
    if remove_index:
        df[column_names[0]] = df.index
        df.reset_index(drop=True, inplace=True)
    columns = df.columns.tolist()
    df = df[[columns[1], columns[0]]]
    return df
    
def graph_seaborn_barplot(data, x, y, hue=None):
    sns.set_theme()
    fig_dims = (15,10)
    fig,ax = plt.subplots(figsize=fig_dims)
    bar = sns.barplot(data=data, x=x, y=y, ax=ax, hue=hue);
    bar.set_xticklabels(bar.get_xticklabels(), rotation=90);
    plt.setp(ax.patches, linewidth=0)

def get_aggregates(match_object: match.MatchData, event):
    events = {
        'bat-runs':('batsmanRuns', 'batsman', 2),
        'bowl-runs':('bowlerRuns', 'bowler', 2),
        'wickets':('isWicket', 'bowler', 1),
        'bat-fours': ('isFour', 'batsman', 1),
        'bowl-fours': ('isFour', 'bowler', 1),
        'bat-sixes':('isSix', 'batsman', 1),
        'bowl-sixes':('isSix', 'bowler', 1),
        'byes':('byes', None, 3),
        'legbyes':('legbyes', None, 3)
    }
    event_mapped = events[event]
    comms = pre_transform_comms(match_object)
    if event_mapped[2] == 1:
        event_s = get_balls_event(comms, column_name=event_mapped[0], value=True)
        event_df = series_to_df(event_s[f'{event_mapped[1]}Name'].value_counts(), [event_mapped[1], event])
    if event_mapped[2] == 2:
        event_df = comms[[f'{event_mapped[1]}Name', event_mapped[0]]]
        event_df = event_df.groupby(f'{event_mapped[1]}Name').sum().sort_values(by=event_mapped[0])
        event_df = series_to_df(event_df, [event_mapped[1], event])
    graph_seaborn_barplot(event_df, event_df.columns[0],event_df.columns[1])

    return event_df

def get_figures_from_scorecard(player_id, _match:match.MatchData, _type, is_object_id=False):
    if not is_object_id: #Change to object ID
        player_id = get_player_map(_match, 'object_id', 'player_id')[player_id]
    url = _match.match_url
    scorecard = wsf.get_match_scorecard(url, match_id=_match.match_id)

    def int_or_none(value):
        try:
            return int(value)
        except ValueError:
            return None

    if _type == 'bowl':
        all_bowlers = [bowler for innings in scorecard for bowler in scorecard[innings]['bowling']]
        
        figures = [f for f in all_bowlers if int(f['bowler'][1]) == int(player_id)]
        bowling_figures = []
        for inning_figures in figures:
            inning_bowling_figures = {
                    'overs': inning_figures['O'],
                    'runs': int_or_none(inning_figures['R']),
                    'dot_balls': 0,
                    'wides': int_or_none(inning_figures['WD']),
                    'noballs': int_or_none(inning_figures['NB'])
                    #need to add wickets, maidens, 4s, 6s, econ
                }
            bowling_figures.append(inning_bowling_figures)
        return bowling_figures
    if _type == 'bat':
        all_batsman = [batsman for innings in scorecard for batsman in scorecard[innings]['batting']]
        
        figures = [f for f in all_batsman if int(f['batsman'][1]) == int(player_id)]
        batting_figures = []
        for inning_figures in figures:
            inning_batting_figures = {
                    'runs': int_or_none(inning_figures['R']),
                    'balls_faced': int_or_none(inning_figures['B']),
                    'fours': int_or_none(inning_figures['4s']),
                    'six': int_or_none(inning_figures['6s']),
                    'dot_balls': 0,
                    'not_out': not bool(inning_figures['out']),
                    'how_out': inning_figures['out']
                }
            batting_figures.append(inning_batting_figures)
        return batting_figures

def check_runout_while_nonstriker(commentary_df:pd.DataFrame, player_id, match_object, is_object_id = False):
    if is_object_id:
        player_id = get_player_map(match_object, 'player_id', 'object_id')[int(player_id)]
    
    wickets = commentary_df[commentary_df.isWicket == True]
    wickets = wickets[wickets.batsmanPlayerId != int(player_id)]
    
    player_name = get_player_map(match_object, 'known_as')[int(player_id)].lower()
    # player_name_split = player_name.split()
    # if len([player_name]) == player_name_split:
    #     player_names = player_name.split('-')
    # else: 
    #     player_names = player_name_split
    
    # contractions = ['ul', 'al', 'du']
    # for contraction in contractions:
    #     try:
    #         player_names.remove(contraction)
    #     except ValueError:
    #         pass

    for i,wicket in wickets.iterrows():
        if player_name in wicket.dismissalText.lower():
            if 'run out' in wicket.dismissalText.lower():
                if int(get_player_team(player_id, match_object)['team']) == int(wicket.battingTeam):
                    return i, wicket
        # if len(wicket.commentTextItems.split()) < 10:
        #     raise utils.NoMatchCommentaryError
        # if any(name in wicket.commentTextItems.lower() for name in player_names):
        #     if int(player_id) != wicket.batsmanPlayerId:
        #         if int(get_player_team(player_id, match_object)['team']) == int(wicket.battingTeam):
        #             return i, wicket
    return None, pd.DataFrame()
    


def get_player_contributions(player_id:str or int, matches:list[match.MatchData], _type = 'both', by_innings = False, is_object_id=False):
    if not isinstance(matches, Iterable):
        matches = [matches]
    
    contributions = []

    for _match in matches:
        if not isinstance(_match, match.MatchData):
            _match = match.MatchData(_match)

        contr_comms = _get_player_contribution(player_id=player_id, _match=_match, _type=_type, by_innings=by_innings, is_object_id=is_object_id)
        if not contr_comms.empty:
            contributions.append(contr_comms)

    return contributions

def _get_player_contribution(player_id:str or int, _match:match.MatchData, _type = 'both', by_innings = False, is_object_id=False):
    """
    Get player innings from a match commentary
    """
    if is_object_id:
        player_id = {player['object_id']:player['player_id'] for player in _match.all_players}[int(player_id)]

    comms = pre_transform_comms(_match)
    # Checking runout from non-strikers end
    i = None
    if _type in ['both', 'bat']:
        i, runout_nonstrikers_df = check_runout_while_nonstriker(comms, player_id, _match, is_object_id=False)

    if _type == 'both':
        comms = comms[(comms['batsmanPlayerId'] == int(player_id)) | (comms['bowlerPlayerId'] == int(player_id))]
    else:
        if _type == 'bat':
            col = 'batsmanPlayerId'
        elif _type == 'bowl':
            col = 'bowlerPlayerId'
        comms = comms[comms[col] == int(player_id)]

    if i:
        comms.loc[i] = runout_nonstrikers_df

    if by_innings:
        comms = [comms[comms['inningNumber'] == i+1] for i, _ in enumerate(_match.innings_list) if not comms[comms['inningNumber'] == i+1].empty]
        # for i, _ in enumerate(match.innings_list):
        #     _comms.append(comms[comms['inningNumber'] == i])
        # _comm
    return comms

def get_cricket_totals(player_id, matches, _type='both', by_innings=False, is_object_id=False, from_scorecards=False):
    if not isinstance(matches, Iterable):
        matches = [matches]
    
    contributions = []

    for _match in matches:
        logger.info('Getting player controbutions for match %s', _match)
        try:
            if not isinstance(_match, match.MatchData):
                _match = match.MatchData(_match)
            contribution = _cricket_totals(player_id, _match, _type, by_innings, is_object_id, from_scorecards=from_scorecards)
            if _type == 'both':
                for i,inning in enumerate(contribution['bat']+contribution['bowl']):
                    contributions.append({**inning, **{key:contribution[key] for key in contribution.keys() if key not in ['bat', 'bowl']}})
                    # contributions.append({**contribution['bowl'], **{key:contribution[key] for key in contribution.keys() if key not in ['bat', 'bowl']}})
            else:
                for i,inning in enumerate(contribution[_type]):
                    contributions.append({**inning, **{key:contribution[key] for key in contribution.keys() if key not in ['bat', 'bowl']}})
        except cricketerrors.MatchNotFoundError:
            logger.warning('Match ID: %s not found', _match)
    return contributions

def _cricket_totals(player_id, m:match.MatchData, _type='both', by_innings=False, is_object_id=False, from_scorecards=False):
    """
    Get the cricketing totals for the players. I.e. their stats in the collected innings.
    """
    if is_object_id: #Change from object id to player id
        player_id = int(get_player_map(m, 'player_id', 'object_id')[int(player_id)])
    batting_figures = None
    bowling_figures = None
    # if player id in bowling id, update bowling figures
    date = m.date
    teams = get_player_team(player_id, m, is_object_id=False)
    team = teams['team']
    opps = teams['opposition']
    continent = m.continent
    ground = m.ground_id

    if _type != 'bat':
        bowling_figures = []
        try:
            if from_scorecards:
                raise utils.NoMatchCommentaryError
            bowling_dfs = _get_player_contribution(player_id, m, 'bowl', by_innings=by_innings, is_object_id=(not is_object_id))
            if not by_innings:
                bowling_dfs = pd.concat(bowling_dfs, ignore_index=True, axis=0)
            if not isinstance(bowling_dfs, list):
                bowling_dfs = [bowling_dfs]
            for bowling_df in bowling_dfs:
                balls_bowled = bowling_df.shape[0]
                bowling_df_agg = bowling_df[['batsmanRuns', 'bowlerRuns', 'noballs', 'wides', 'isSix', 'isFour', 'isWicket', 'legbyes']].sum(numeric_only=False)
                extras = bowling_df_agg['wides']+bowling_df_agg['noballs']
                inning = bowling_df['inningNumber'].iloc[0]
                inning_bowling_figures = {
                    'inning':inning,
                    'overs': f'{(balls_bowled - extras)//6}.{(balls_bowled - extras)%6}',
                    'runs': bowling_df_agg['bowlerRuns'],
                    'dot_balls': (bowling_df['bowlerRuns'] == 0).sum(),
                    'wides': bowling_df_agg['wides'],
                    'noballs': bowling_df_agg['noballs']
                }
                bowling_figures.append(inning_bowling_figures)
        except utils.NoMatchCommentaryError:
            logger.info("Getting bowling figures from scorecard")
            bowling_figures += get_figures_from_scorecard(player_id, m, 'bowl', is_object_id=(not is_object_id))
    # else if player id in batting id, update batting figures
    if _type != 'bowl':
        batting_figures = []
        try:
            if from_scorecards:
                raise utils.NoMatchCommentaryError
            batting_dfs = _get_player_contribution(player_id, m, 'bat', by_innings=by_innings, is_object_id=(not is_object_id))
            if not by_innings:
                batting_dfs = pd.concat(batting_dfs, ignore_index=True, axis=0)
            if not isinstance(batting_dfs, list):
                batting_dfs = [batting_dfs]
            for batting_df in batting_dfs:
                not_out = not batting_df['isWicket'].iloc[-1]
                balls_faced = batting_df[batting_df.batsmanPlayerId == player_id].shape[0]
                batting_df_agg = batting_df[batting_df.batsmanPlayerId == player_id].sum()
                inning = batting_df['inningNumber'].iloc[0]
                inning_batting_figures = {
                    'inning': inning,
                    'runs': batting_df_agg['batsmanRuns'],
                    'balls_faced': balls_faced,
                    'fours': batting_df_agg['isFour'],
                    'six': batting_df_agg['isSix'],
                    'dot_balls': (batting_df[batting_df.batsmanPlayerId == player_id]['bowlerRuns'] == 0).sum(),
                    'not_out': not_out,
                    'how_out': None
                }
                batting_figures.append(inning_batting_figures)
        except utils.NoMatchCommentaryError:
            logger.info("Getting batting figures from scorecard")
            batting_figures += get_figures_from_scorecard(player_id, m, 'bat', is_object_id=(not is_object_id))
        logger.debug("Match ID: %s\nBatting: %s\nBowling: %s",m.match_id, batting_figures, bowling_figures) 
    return {'bat': batting_figures, 'bowl': bowling_figures, 'date':datetime.strptime(date, "%Y-%m-%d"),'team':team, 'opposition': opps, 'ground':ground, 'continent':continent}

def process_text_comms(df:pd.DataFrame, columns = ['dismissalText', 'commentPreTextItems', 'commentTextItems', 'commentPostTextItems', 'commentVideos']):
    for column in columns:
        df[column] = df[column].map(process_text_values)
        df[column] = df[column].map(remove_html)

def remove_html(value):
    try:
        matches = re.findall('[^<>]{0,}(<[^<>]+>)[^<>]{0,}', value)
        for match in matches:
            value = value.replace(match, '')
        value = value.capitalize()
    except TypeError:
        pass
    finally:
        return value

def process_text_values(value):
    if not isinstance(value, list):
        value = [value]
    try:
        value = value[0]
        if 'type' in value.keys():
            if value['type'] == 'HTML':
                return value['html']
        else:
            return value['commentary']
    except (TypeError, KeyError, AttributeError):
        if value is None:
            return 'None'
        # print('Unhandled value', value)
        pass
    except IndexError:
        return 'None'

def create_dictionary(words, reverse = False):
    dictionary = {}
    for word in words:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse= not reverse))

def create_vocabulary(df, m, remove_match_refs = True):
    player_list = list(get_player_map(m, map_column='known_as').values())
    player_list = [name.lower() for player in player_list for name in player.split()]

    words = []
    for entry in df['commentTextItems']:
        entry = entry.translate(str.maketrans('', '', string.punctuation))
        entry = entry.split()
        if remove_match_refs:
            words += [e.lower() for e in entry if e not in player_list]
        else:
            words += [e.lower() for e in entry]

    dictionary = create_dictionary(words)
    return dictionary

def create_dummies(df: pd.DataFrame, column = 'batsmanRuns', value_mapping = {'isRuns': [1,2,3,5]}):
    for value in value_mapping:
        df[value] = df[column].isin(value_mapping[value])

def create_labels(df: pd.DataFrame, categories:list, null_category:str, rev_dummy_col_name:str = 'labels', commentary_col = 'commentTextItems', inplace = False):
    if not inplace:
        result = df.loc[:, categories+[commentary_col]]
    else:

        result = df
    logger.info(f'Creating labels for commentary. \n \
                 Columns to label: {", ".join(categories)} \n \
                 Commentary text column: {commentary_col}')
    result.loc[:, null_category] = False
    categories += [null_category]
    logger.debug('Reverse dummy to create labels column')
    for index, row in result.iterrows():
        if int(row.loc[categories].max()) == 0:
            result.at[index, null_category] = True
    reverse_dummy = result[categories].idxmax(axis=1)
    reverse_dummy = reverse_dummy.to_frame(name= rev_dummy_col_name)
    logger.debug('Creating label dataframe')
    if not inplace:
        #print(result.head())
        reverse_dummy[commentary_col] = result[commentary_col]
        return reverse_dummy
    else:
        result[rev_dummy_col_name] = reverse_dummy[rev_dummy_col_name]
        return result

def cat_to_num(labels, label_names):
    mapping = {cat:i for i,cat in enumerate(label_names)}
    labels_nums = [mapping[label] for label in labels]
    return labels_nums

def package_data(data:list, labels:list, label_names:list = [], encode_num = True):
    if not label_names:
        label_names = set(labels)
        label_names = list(label_names)
    if encode_num:
        labels = np.array(cat_to_num(labels, label_names))
    packaged_data = sklearn.utils.Bunch(
        data = list(data), labels = labels, label_names = list(label_names)
    )
    return packaged_data

def describe_data_set(dataset, title, label_names=None):
    if isinstance(dataset, sklearn.utils.Bunch):
        labels_unmapped = [dataset.label_names[label] for label in dataset.labels]
    else:
        labels_unmapped = [label_names[label] for label in dataset]
    series = pd.Series(labels_unmapped, )
    groups = series_to_df(series, [title,'labels']).groupby('labels').count()/series.shape[0]
    return groups

def calculate_running_average(innings_df):
    if innings_df.empty:
        logger.info('No innings for player')
        return None

    _running_average = []
    innings_df = innings_df[innings_df.runs.notna()]
    total_runs = 0
    out = 0

    for innings in zip(innings_df.runs, innings_df.not_out):
        total_runs += innings[0]
        if innings[1] == False:
            out += 1
        try:
            _running_average.append(round(total_runs/out,2))
        except ZeroDivisionError:
            _running_average.append(None)

    return _running_average

def calculate_recent_form_average(innings_df:pd.DataFrame, window_size=12):
    if innings_df.empty:
        logger.info('No innings for player')
        return None

    last_x_average = []
    innings_df = innings_df[innings_df.runs.notna()]
    window_runs = 0
    window_out = 0

    for i,innings in enumerate(zip(innings_df.runs, innings_df.not_out)):
        if i>=window_size:
            window_runs -= innings_df.runs.iloc[i-window_size]
            if innings_df.not_out.iloc[i-window_size] == False:
                window_out -= 1
        
        window_runs += innings[0]
        if innings[1] == False:
            window_out += 1
        try:
            last_x_average.append(round(window_runs/window_out,2))
        except ZeroDivisionError:
            last_x_average.append(None)

    return last_x_average

def get_running_average(player_id, innings = None, match_list=None, _format='test'):
    if innings is None:
        if match_list is None:
            match_list = wsf.player_match_list(player_id, _format=_format, match_links=False)
        innings = get_cricket_totals(player_id, match_list, _type='bat', by_innings=True, is_object_id=True)
    #innings = [inning for match in contributions for inning in match]
    innings_df = pd.DataFrame(innings)
    average = calculate_running_average(innings_df)
    return average

def get_recent_form_average(player_id, innings=None, match_list=None, window_size=10,_format='test'):
    if innings is None:
        if match_list is None:
            match_list = wsf.player_match_list(player_id, _format=_format, match_links=False)
        innings = get_cricket_totals(player_id, match_list, _type='bat', by_innings=True, is_object_id=True)
    # innings = [inning for match in contributions for inning in match]
    innings_df = pd.DataFrame(innings)
    average = calculate_recent_form_average(innings_df, window_size=window_size)
    return average

def get_player_team(player_id, _match:match.MatchData, is_object_id=False):
    if is_object_id:
        map_id = 'object_id'
    else:
        map_id = 'player_id'

    team_1 = [int(player[map_id]) for player in _match.team_1_players]
    team_2 = [int(player[map_id]) for player in _match.team_2_players]
    
    if int(player_id) in team_1:
        return {'team': _match.team_1_id, 'opposition': _match.team_2_id}
    elif int(player_id) in team_2:
        return {'team': _match.team_2_id, 'opposition': _match.team_1_id}
    else:
        raise utils.PlayerNotPartOfMatch('Player not part of match')

def percentagize_x_axis(data, _round = 2):
    length = len(data)
    return [round(x*(100/length), _round) for x in range(length)]

def normalized_career_length(career_data:dict):
    #recent_form_df = {}
    #max_length = career_data.index(max(career_data, key=len))
    #index = set([i for data in career_data for i in percentagize_x_axis(data)])
    #index = sorted(index)
    full_df = pd.concat([pd.DataFrame(career_data[data], index=percentagize_x_axis(career_data[data]), columns=[data]) for data in career_data], axis=1)
    full_df.sort_index(inplace=True)
    return full_df

def apply_aggregate_func_to_list(player_id_list, _funcs, player_ages=None, return_innings = False, disable_logging=True, **kwargs):
    if player_ages:
        if not isinstance(player_ages, list):
            player_ages = [player_ages]
        if len(player_ages) == 1:
            player_ages = player_ages*len(player_id_list)
        if len(player_ages) != len(player_id_list):
            player_ages = player_ages + [':']*abs(len(player_ages) != len(player_id_list))
            player_ages = player_ages[:len(player_id_list)]
    else:
        player_ages = [None for i in range(len(player_id_list))]
    logger.disabled = disable_logging
    applied_values = defaultdict(dict)
    for i,player in enumerate(player_id_list):
        dates = dates_from_age(player, player_ages[i])
        player_match_list = wsf.get_player_match_list(player, dates=dates)
        player_innings_df = get_cricket_totals(player, player_match_list, _type='bat', by_innings=True, is_object_id=True)
        player_innings_df = pd.DataFrame(player_innings_df)
        for _func in _funcs:
            function_name = re.search('function ([\_a-zA-Z]+) at', str(_func)).group(1)
            applied_values[function_name][player] = _func(player_innings_df, **kwargs)
        if return_innings:
            applied_values['innings_totals'][player] = player_innings_df
    if disable_logging:
        logger.disabled = not disable_logging
    
    return applied_values