import requests
from espncricinfo.match import Match

DETAILED_COMMS_BASE_URL = 'https://hs-consumer-api.espncricinfo.com/v1/pages/match/comments?lang=en&seriesId={seriesid}&matchId={matchid}&inningNumber={inning}&commentType=ALL&sortDirection=DESC'

class MatchData(Match):

    def __init__(self, match_id):
        super().__init__(match_id)
        self.detailed_comms_url = DETAILED_COMMS_BASE_URL.replace('{seriesid}', str(self.series_id)).replace('{matchid}', str(self.match_id))
        self.full_comms = self.get_detailed_comms()
        self.first_inning = self.get_innings_comms(innings = 1)
        self.second_innings = self.get_innings_comms(innings = 2)
        self.third_innings = self.get_innings_comms(innings = 3)
        self.fourth_innings = self.get_innings_comms(innings = 4)

    def get_detailed_comms(self):
        """
        Detailed commentary for the whole match
        """
        try:
            full_comms = []
            for innings in self.innings_list:
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