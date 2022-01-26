import os
from asyncio import as_completed
import requests
from espncricinfo.match import Match
from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
import json
from codebase.settings import DATA_LOCATION

DETAILED_COMMS_BASE_URL = 'https://hs-consumer-api.espncricinfo.com/v1/pages/match/comments?lang=en&seriesId={seriesid}&matchId={matchid}&inningNumber={inning}&commentType=ALL&sortDirection=DESC'

class MatchData(Match):

    def __init__(self, match_id, try_local = True):
        super().__init__(match_id)
        self.detailed_comms_url = DETAILED_COMMS_BASE_URL.replace('{seriesid}', str(self.series_id)).replace('{matchid}', str(self.match_id))
        self.full_comms = self.get_detailed_comms_faster(try_local = try_local)
        self.first_inning = self.get_innings_comms(innings = 1)
        self.second_innings = self.get_innings_comms(innings = 2)
        self.third_innings = self.get_innings_comms(innings = 3)
        self.fourth_innings = self.get_innings_comms(innings = 4)

    def get_detailed_comms_faster(self, try_local=True, save=True):
        """
        Detailed commentary for the whole match
        """
        try:
            if try_local:
                if os.path.exists(os.path.join(DATA_LOCATION, f'{self.match_id}_full_comms.json')):
                    with open(os.path.join(DATA_LOCATION, f'{self.match_id}_full_comms.json'), 'r') as jf:
                        return json.load(jf)
            full_comms = []
            for innings in self.innings_list: ##Sessionize innings search and parallize overs.
                INNINGS_URL = self.detailed_comms_url.replace('{inning}', str(innings['innings_number']))
                with FuturesSession() as inning_session:
                    print(f'Fetching inning {innings["innings_number"]} comms')
                    init_future = inning_session.get(INNINGS_URL)
                    init_response = init_future.result()
                    overs = init_response.json()['nextInningOver'] 
                    if overs == 'null' or overs is None:
                        continue
                    
                    full_comms += init_response.json()['comments']
                    concurrent_requests = int(overs)//2
                    
                    innings_comms = [inning_session.get(INNINGS_URL+f'&fromInningOver={(x+1)*2}') for x in reversed(range(concurrent_requests))]
                    for chunk in innings_comms:
                        full_comms += chunk.result().json()['comments']
                    print('Innings completed')
            if save:
                self.save_json_data(full_comms, 'full_comms')
            return full_comms
        except:
            print("Init Response Innings: \n",init_response)
            print("Innings: \n", innings_comms)


    def save_json_data(self, data, suffix):
        with open(os.path.join(DATA_LOCATION, f'{self.match_id}_{suffix}.json'), 'w') as f:
            f.write(json.dumps(data))

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
        pass


if __name__ == '__main__':
    pass