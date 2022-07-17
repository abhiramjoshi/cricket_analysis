import codebase.analysis_functions as af
import codebase.web_scrape_functions as wsf
from codebase.match_data import MatchData
import pandas as pd
from utils import logger
import utils
import matplotlib.pyplot as plt
import seaborn as sns

def graph_multi_player_batting_careers(player_ids, dates=None, player_ages = None, graph_elements = [True, True, True], disable_logging=False):
    """
    Graph Batting Careers for multiple players

    player_ids: List of player_ids
    dates: The dates between which the careers should be graphed, date format YYYY-MM-DD:YYYY-MM-DD
    player_ages: List of player ages that you want to graph the data from, if you define a single number then all players will be graphed after that age
    graph_elements: Elements of the batting graph to plot [recent form, running average, innings bars]
    """
    #need to get player forms, and then need to graph it in a subplot
    keys = ['calculate_recent_form_average', 'calculate_running_average', 'innings_totals']
    
    _funcs = []
    if graph_elements[0]:
        _funcs.append(af.calculate_recent_form_average)
    if graph_elements[1]:
        _funcs.append(af.calculate_running_average)
    all_stats = af.apply_aggregate_func_to_list(player_ids, _funcs=_funcs, dates=dates, player_ages=player_ages, return_innings=graph_elements[2], disable_logging=disable_logging)

    for key in keys:
        try:
            all_stats[key]
        except KeyError:
            all_stats[key] = None

    graph_career_batting_summary(
        recent_form=all_stats[keys[0]], 
        running_ave=all_stats[keys[1]], 
        innings_scores=all_stats[keys[2]],
    )



def graph_career_batting_summary(recent_form=None, running_ave=None, innings_scores=None, x_label = None, y_label = None, barhue=None):
    combined_averages = {**{k:recent_form[k] for k in sorted(recent_form)}, **{f'{key}_rf':running_ave[key] for key in sorted(running_ave)}}
    k = len(combined_averages)//2
    fig, ax1 = plt.subplots(nrows=k, figsize=(18, k*5))
    sns.set_theme()


    for i in range(k):
        first_column = list(combined_averages.keys())[i]
        second_column = list(combined_averages.keys())[i+k]
        logger.info(f"Graphing career for player: {first_column}")

        if innings_scores:
            logger.debug('Graphing inning by inning scores')
            try:
                y_range = max([0, max(innings_scores[first_column].runs) + 20], [0, max(combined_averages[first_column]) + 20])
            except ValueError:
                if not combined_averages[first_column]:
                    logger.info('Player %s has no matches to graph', first_column)
                    continue
                y_range = [0, max(combined_averages[first_column]) + 20]

            ax2 = ax1[i].twinx()
            ax2, ax1[i] = ax1[i], ax2
        
        logger.debug("Graphing career agreggate averages")
        sns.lineplot(data = {'recent form':combined_averages[first_column], 'career ave':combined_averages[second_column]}, sort = False, ax=ax1[i], palette='rocket', lw=2, zorder=0)
        name = wsf.get_player_json(first_column)["name"]
        ax1[i].set_title(f'{name} Career Summary')
        if x_label:
            ax1[i].set_xlabel(x_label)
        if y_label:
            ax1[i].set_ylabel(y_label)
        
        if innings_scores:
            if barhue is not None:
                barhue = innings_scores[first_column].barhue
            
            sns.barplot(data = innings_scores[first_column], x=innings_scores[first_column].index, y=innings_scores[first_column].runs, alpha=0.8, ax=ax2, hue=barhue, palette='mako', dodge=False, zorder=10)
            ax2.set_xticklabels(labels=innings_scores[first_column].index, rotation=90);
            ax2.xaxis.set_major_locator(plt.MaxNLocator(15))
            ax1[i].set_ylim(y_range)
            ax2.set_ylim(y_range)
    
    if not utils.check_if_ipython():
        plt.show()


def get_career_batting_graph(player_id:str or int, _format:str = 'test', player_age=None, dates:str=None, barhue:str=None, window_size:int = 12, label_spacing=10):
    """
    Gets player contributions between the dates provided and graphs the innings, running average and form average
    NOTE: player_id is object_id.

    player-age: See career graph based on a segement of the players age. Format younger age:older age. 
    """
    if player_age:
        dates = af.dates_from_age(player_id, player_age)
        

    logger.info('Getting match list for player, %s', player_id)
    match_list = wsf.player_match_list(player_id, dates=dates, _format=_format)
    logger.info('Getting player contributions for %s', player_id)
    innings = af.get_cricket_totals(player_id, match_list, _type='bat', by_innings=True, is_object_id=True)
    innings_df = pd.DataFrame(innings)
    logger.info('Calculating running average for %s', player_id)
    running_av = af.get_running_average(player_id,innings=innings)
    logger.info('Calculating recent form average with window size %s for %s', window_size, player_id)
    recent_form = af.get_recent_form_average(player_id, innings=innings, window_size=window_size)

    if barhue is not None:
        barhue = innings_df.barhue

    logger.info("Plotting career batting summary")
    y_range = [0, max(innings_df.runs) + 20]

    fig, ax1 = plt.subplots(figsize=(18,10)) 
    #sns.set_theme()
    sns.barplot(data = innings_df, x=innings_df.index, y=innings_df.runs, alpha=0.8, ax=ax1, hue=barhue, palette='mako', dodge=False)

    ax1.set_ylim(y_range)

    ax2 = ax1.twinx()

    sns.lineplot(data = {'Average': running_av, f'Last {window_size} Innings': recent_form}, sort = False, ax=ax2, palette='rocket', linewidth=2)
    ax2.set_ylim(y_range)
    x_dates = innings_df.date.dt.strftime('%d-%m-%Y')
    ax1.set_xticklabels(labels=x_dates, rotation=90);
    ax1.xaxis.set_major_locator(plt.MaxNLocator(label_spacing))
    ax1.margins(x=0)