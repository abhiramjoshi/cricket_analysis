from codebase.settings import LABEL_DATA
import os
from pprint import pprint

FILE = '/Users/abhiramjoshi/Documents/Personal/cricket_analysis/data/drive_dismissals_kohli.txt'
PROMPT = 'Example of cover drive dismissal?'
OPTIONS = [
    "Yes",
    "No"
]
OPTIONS = {i:key for i,key in enumerate(OPTIONS)}

SAVEFILE = os.path.join(LABEL_DATA, f'labelled_{FILE}')

labels = []
print(f'Reading file from {FILE}\n---------------------\n')

with open(FILE, 'r') as file:
    for line in file.readlines():
        print(line)
        print(f'{PROMPT}\nSelect the number for the valid option:')
        pprint(OPTIONS)
        label = int(input('Select an option: '))
        labels.append((line,  OPTIONS[label]))

print('All samples have been labelled')

with open(f'labelled_{FILE}', 'w') as outfile:
    for line in labels:
        outfile.writelines(line)


