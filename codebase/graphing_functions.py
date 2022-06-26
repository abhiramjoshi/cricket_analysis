import matplotlib
import codebase.analysis_functions as af
import codebase.web_scrape_functions as wsf
from codebase.match_data import MatchData
import pandas as pd
from utils import logger
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib
from codebase.settings import DATA_LOCATION
import os

matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe"

def graph_career_batting_summary(recent_form=None, running_ave=None, innings_scores=None, x_label = None, y_label = None, barhue=None):
    combined_averages = {**{k:recent_form[k] for k in sorted(recent_form)}, **{f'{key}_rf':running_ave[key] for key in sorted(running_ave)}}
    k = len(combined_averages)//2
    fig, ax1 = plt.subplots(nrows=k, figsize=(18, k*5), sharey=True)
    sns.set_theme()
    for i in range(k):
        first_column = list(combined_averages.keys())[i]
        second_column = list(combined_averages.keys())[i+k]
        logger.info(f"Graphing career for player: {first_column}")

        if innings_scores:
            logger.debug('Graphing inning by inning scores')
            try:
                y_range = [0, max(innings_scores[first_column].runs) + 20]
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

def get_animated_career(player_id:str or int, _format:str = 'test', player_age=None, dates:str=None, barhue:str=None, window_size:int = 12, label_spacing=10):
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
    ax1.set_ylim(y_range)
    ax2 = ax1.twinx()
    ax2.set_ylim(y_range)
    sns.barplot(data = innings_df, x=innings_df.index, y=innings_df.runs, alpha=0.8, ax=ax1, hue=barhue, palette='mako', dodge=False)
    x_dates = innings_df.date.dt.strftime('%d-%m-%Y')
    ax1.set_xticklabels(labels=x_dates, rotation=90);
    ax1.xaxis.set_major_locator(plt.MaxNLocator(max(innings_df.shape[0]//5, 2)))
    ax1.margins(x=0)
    def animate(i):
        #innings_df_part = innings_df.iloc[:(i+1)]
        running_ave_part = running_av[:(i+1)]
        recent_form_part = recent_form[:(i+1)]
        sns.lineplot(data = {'Average': running_ave_part, f'Last {window_size} Innings': recent_form_part}, sort = False, ax=ax2, palette='rocket', linewidth=2, legend=False)

    logger.info('Animating...')
    ani = FuncAnimation(fig, animate, frames=innings_df.shape[0])
    writer = animation.FFMpegWriter(fps=15)
    logger.info('Saving animation to %s', os.path.join(DATA_LOCATION, 'animation_test.mp4'))
    ani.save(os.path.join(DATA_LOCATION, 'animation_test.mp4'), writer=writer, )

if __name__ == "__main__":
    get_animated_career('253802')