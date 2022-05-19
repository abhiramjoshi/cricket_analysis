import codebase.match_data as match
import numpy as np
import pandas as pd
import sklearn.utils
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import utils
from utils import logger

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
    return df

def get_balls_event(comms:pd.DataFrame, column_name:str, value, negative=False):
    if negative:
        event_df = comms[comms[column_name] != value]    
    else:
        event_df = comms[comms[column_name] == value]
    return event_df

def get_player_map(match_object:match.MatchData, name_tag:str='card_long'):
    return {int(player['player_id']):player[name_tag] for player in match_object.all_players}

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
    
def graph_seaborn_barplot(data, x, y):
    sns.set_theme()
    fig_dims = (15,10)
    fig,ax = plt.subplots(figsize=fig_dims)
    bar = sns.barplot(data=data, x=x, y=y, palette='Blues_d', ax=ax);
    bar.set_xticklabels(bar.get_xticklabels(), rotation=90);

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

def get_player_contribution(player_id:str|int, match:match.MatchData, _type = 'both', by_innings = False, is_object_id=False):
    """
    Get player innings from a match commentary
    """
    if is_object_id:
        player_id = {player['object_id']:player['player_id'] for player in match.all_players}[int(player_id)]

    comms = pre_transform_comms(match)
    if _type == 'both':
        comms = comms[(comms['batsmanPlayerId'] == int(player_id)) | (comms['bowlerPlayerId'] == int(player_id))]
    else:
        if _type == 'bat':
            col = 'batsmanPlayerId'
        elif _type == 'bowl':
            col = 'bowlerPlayerId'
        comms = comms[comms[col] == int(player_id)]
        if by_innings:
            comms = [comms[comms['inningNumber'] == i] for i, _ in enumerate(match.innings_list)]
            # for i, _ in enumerate(match.innings_list):
            #     _comms.append(comms[comms['inningNumber'] == i])
            # _comm
    return comms

def cricket_totals(player_id, m, by_innings=False, is_object_id=False):
    """
    Get the cricketing totals for the players. I.e. their stats in the collected innings.
    """
    # if player id in bowling id, update bowling figures
    bowling_dfs = get_player_contribution(player_id, m, 'bowl', by_innings=by_innings, is_object_id=is_object_id)
    bowling_figures = []
    if not by_innings:
        bowling_dfs = pd.concat(bowling_dfs, ignore_index=True, axis=0)
    if not isinstance(bowling_dfs, list):
        bowling_dfs = [bowling_dfs]
    for bowling_df in bowling_dfs:
        balls_bowled = bowling_df.shape[0]
        bowling_df_agg = bowling_df[['batsmanRuns', 'bowlerRuns', 'noballs', 'wides', 'isSix', 'isFour', 'isWicket', 'legbyes']].sum(numeric_only=False)
        extras = bowling_df_agg['wides']+bowling_df_agg['noballs']
        inning_bowling_figures = {
            'overs': f'{(balls_bowled - extras)//6}.{(balls_bowled - extras)%6}',
            'runs': bowling_df_agg['bowlerRuns'],
            'dot_balls': (bowling_df['bowlerRuns'] == 0).sum(),
            'wides': bowling_df_agg['wides'],
            'noballs': bowling_df_agg['noballs']
        }
        bowling_figures.append(inning_bowling_figures)
    # else if player id in batting id, update batting figures
    batting_dfs = get_player_contribution(player_id, m, 'bat', by_innings=by_innings, is_object_id=is_object_id)
    batting_figures = []
    if not by_innings:
        batting_dfs = pd.concat(batting_dfs, ignore_index=True, axis=0)
    if not isinstance(batting_dfs, list):
        batting_dfs = [batting_dfs]
    for batting_df in batting_dfs:
        balls_faced = batting_df.shape[0]
        batting_df_agg = batting_df.sum()
        inning_batting_figures = {
            'runs': batting_df_agg['batsmanRuns'],
            'balls_faced': balls_faced,
            'fours': batting_df_agg['isFour'],
            'six': batting_df_agg['isSix'],
            'dot_balls': (batting_df['bowlerRuns'] == 0).sum()
        }
        batting_figures.append(inning_batting_figures)
    return batting_figures, bowling_figures

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
    player_list = list(get_player_map(m, name_tag='known_as').values())
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
