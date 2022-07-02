import requests
from espncricinfo.match import Match
from requests_futures.sessions import FuturesSession
from codebase.settings import DATA_LOCATION, FULL_COMMS
from utils import logger

DETAILED_COMMS_BASE_URL = 'https://hs-consumer-api.espncricinfo.com/v1/pages/match/comments?lang=en&seriesId={seriesid}&matchId={matchid}&inningNumber={inning}&commentType=ALL&sortDirection=DESC'

class MatchData(Match):

    def __init__(self, match_id, try_local=True, serialize=False, save=True):
        super().__init__(int(match_id), try_local, serialize, save)
        self.detailed_comms_url = DETAILED_COMMS_BASE_URL.replace('{seriesid}', str(self.series_id)).replace('{matchid}', str(self.match_id))
        self.full_comms = self.get_detailed_comms_faster(try_local = try_local, save=save, serialize=serialize)
        self.all_players = self.team_1_players + self.team_2_players
        self.first_inning = self.get_innings_comms(innings = 1)
        self.second_innings = self.get_innings_comms(innings = 2)
        self.third_innings = self.get_innings_comms(innings = 3)
        self.fourth_innings = self.get_innings_comms(innings = 4)
        if not self.innings_list:
            self.innings_list = self.innings

    def get_detailed_comms_faster(self, try_local=True, save=True, serialize=True):
        """
        Detailed commentary for the whole match
        """
        logger.info(f'Match ID {self.match_id}: Loading match comms')
        try:
            if try_local:
                data = self.load_data(suffix='full_comms', data_folder=FULL_COMMS)
                if data:
                    return data
                # if os.path.exists(os.path.join(DATA_LOCATION, f'{self.match_id}_full_comms.json')):
                #     logger.info(f'{self.match_id}: Loading commentary from local storage')
                #     with open(os.path.join(DATA_LOCATION, f'{self.match_id}_full_comms.json'), 'r') as jf:
                #         return json.load(jf)


            logger.info(f'{self.match_id}: Loading commentary from Cricinfo')
            full_comms = []

            for innings in self.innings_list: ##Sessionize innings search and parallize overs.
                innings_comm_buffer = []
                INNINGS_URL = self.detailed_comms_url.replace('{inning}', str(innings['innings_number']))
                with FuturesSession() as inning_session:
                    logger.info(f'Fetching inning {innings["innings_number"]} commentary')
                    logger.debug(f'Requesting commentary from: {INNINGS_URL}')
                    init_future = inning_session.get(INNINGS_URL)
                    init_response = init_future.result()
                    overs = init_response.json()['nextInningOver'] 
                    if overs == 'null' or overs is None:
                        innings_comm_buffer += init_response.json()['comments']
                        full_comms.append(list(reversed(innings_comm_buffer)))
                        continue
                    
                    innings_comm_buffer += init_response.json()['comments']
                    concurrent_requests = int(overs)//2
                    
                    innings_comms_requests = [inning_session.get(INNINGS_URL+f'&fromInningOver={(x+1)*2}') for x in reversed(range(concurrent_requests))]
                    for chunk in innings_comms_requests:
                        logger.debug(f'Requesting commentary from: {chunk}')
                        innings_comm_buffer += chunk.result().json()['comments']
                    logger.info('Innings completed')
                    full_comms.append(list(reversed(innings_comm_buffer)))
            if save:
                self.save_data(data=full_comms, suffix='full_comms', serialize=serialize, data_folder=FULL_COMMS)
            return full_comms
        except TypeError as e:
            logger.warning(f'Match {self.match_id}, {self.series_name} - {self.match_title} does not have commentary data')
        except IndexError as e:
            logger.warning("Init Response Innings: \n",init_response)
            logger.warning("Innings: \n", innings_comms_requests)

    def get_detailed_comms(self):
        """
        Detailed commentary for the whole match
        """
        try:
            full_comms = []
            for innings in self.innings_list: ##Sessionize innings search and parallize overs.
                INNINGS_URL = self.detailed_comms_url.replace('{inning}', str(innings['innings_number']))
                OVER_URL = ''
                print(f'Fetching inning {innings["innings_number"]} comms')
                while True:
                    URL = INNINGS_URL+OVER_URL
                    response = requests.get(URL).json()
                    full_comms += response['comments']
                    if response['nextInningOver'] == 'null' or response['nextInningOver'] == None:
                        break
                    OVER_URL = f'&fromInningOver={response["nextInningOver"]}'
                print('Innings completed')
            return full_comms
        except:
            print(response)

    def get_innings_comms(self, innings):
        try:
            return self.full_comms[innings-1]
        except (IndexError, TypeError):
            return None

    def get_full_comms(self):
        """
        Flatten and return full comms
        """
        if self.full_comms is None or not self.full_comms:
            return None
        if not isinstance(self.full_comms[0], list):
            return list(reversed([ball for ball in self.full_comms]))
        return [ball for innings in self.full_comms for ball in innings]
       
if __name__ == '__main__':
    pass