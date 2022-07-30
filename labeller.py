from codebase.settings import LABEL_DATA
import os
from pprint import pprint

FILE = 'D:\Documents\Cricket\data_analysis\cricket_data_analysis\data\drive_dismisals.txt'
PROMPT = 'Example of cover drive dismissal?'
OPTIONS = [
    "Yes",
    "No"
]
OPTIONS = {i:key for i,key in enumerate(OPTIONS)}

selections = ['0', '1', '0', '1', '0', '0', '0', '1', '1', '0', '0', '0', '1', '0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '0', '1', '0', '1', '1', '0', '1', '0', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '1', '0', '1', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0']

SAVEFILE = os.path.join(LABEL_DATA, f'labelled_drive_dismisals.txt')

labels = []
print(f'Reading file from {FILE}\n---------------------\n')

try:
    with open(FILE, 'r') as file:
        for i,line in enumerate(file.readlines()):
            line = line.replace('\n', '').strip(',')
            #print(line)
            #print(f'{PROMPT}\nSelect the number for the valid option:')
            #pprint(OPTIONS)
            #print()
            #label = int(input('Select an option: '))
            #labels.append((line,  OPTIONS[label]))
            print((line,  OPTIONS[int(selections[i])]))
            labels.append((line,  OPTIONS[int(selections[i])]))

    print('All samples have been labelled')

except KeyboardInterrupt:
    if input('Continue with saving? y/N: ').lower() == 'y':
        pass
    else:
        print('Quitting')
        quit()

with open(SAVEFILE, 'w+') as outfile:
    for line in labels:
        outfile.write(f'{line[0]}, label: {line[1]}\n')


