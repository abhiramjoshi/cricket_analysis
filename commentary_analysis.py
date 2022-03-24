import pickle
import sys
from turtle import pen
from matplotlib.pyplot import vlines
import sklearn
from sklearn.linear_model import SGDClassifier
import codebase.match_data as match
import codebase.settings as settings
import pandas as pd
import numpy as np
import os
from pprint import pprint
import codebase.analysis_functions as af
import utils
import codebase.web_scrape_functions as wsf
import numpy as np
from codebase.match_data import MatchData
from utils import logger
from codebase.settings import DATA_LOCATION, ANALYSIS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

years = ['2017', '2021']
LABEL_DATA_FILENAME= f"train_commentary_labels_{years[0]}_{years[-1]}.json"
FORCE_LABEL = False
FORCE_BUNCH = True
SHUFFLE = True
TEST_TRAINING_SPLIT = 0.9

if __name__ == "__main__":
    all_comms = []

    if not os.path.exists(os.path.join(ANALYSIS, LABEL_DATA_FILENAME)) or FORCE_LABEL:
        matchlist = wsf.get_match_list(years=['2017', '2021'], finished=True)
        logger.info(f"Number of matches selected: \n{len(matchlist)}")
        e = input('Continue (y/N): ')
        if e.lower() != 'y':
            sys.exit()  
        # matchlist = [x[1] for x in matchlist]
        
        for m_id in matchlist:
            try:
                logger.info(f'Grabbing data for matchID {m_id}')
                _match = MatchData(m_id, serialize=False)
                comms = af.pre_transform_comms(_match)
                comm_w_labels = af.create_labels(comms, ['isWicket', 'isFour', 'isSix'], null_category='noEvent')
                all_comms.append(comm_w_labels)
            except utils.NoMatchCommentaryError:
                continue


        try:
            all_comms = pd.concat(all_comms, ignore_index=True)
            logger.info('All commentary dataframe stats')
            logger.info(all_comms.size)
            logger.info(all_comms.groupby('labels').size())
            logger.info('Saving labelled commentary to JSON')
            all_comms.to_json(os.path.join(ANALYSIS, LABEL_DATA_FILENAME))
            logger.info(f'Commentary saved to {os.path.join(ANALYSIS, LABEL_DATA_FILENAME)}')
        except ValueError:
            print('No commentary to show')

    else:
        all_comms = pd.read_json(os.path.join(ANALYSIS, LABEL_DATA_FILENAME))

    training_set_location = os.path.join(ANALYSIS, 'training_set_bunch.p')

    if os.path.exists(training_set_location) or FORCE_BUNCH:
        logger.info(f'Skipping data processing as labelled commentary already exists at {LABEL_DATA_FILENAME}')
        if SHUFFLE:
            logger.info("Shuffling dataset before test/train split")
            all_comms = all_comms.sample(frac=1)
        n = all_comms.shape[0]
        data = all_comms['commentTextItems']
        labels = all_comms['labels']
        logger.info(all_comms.groupby('labels').count()/all_comms.shape[0])
        logger.info('Creating training set with %s of the data', TEST_TRAINING_SPLIT)
        split = int(n*TEST_TRAINING_SPLIT)
        training_set = af.package_data(data=data[:split], labels=labels[:split])
        logger.info('Creating test set')
        test_set = af.package_data(data=data[:(len(data)-split)], labels=labels[:(len(data)-split)], label_names=training_set.label_names)
    # pprint(training_set.data[-10:])
    # pprint(training_set.labels[-10:])
    # p
    # print(training_set.label_names)

        with open(training_set_location, 'wb') as tr:
            pickle.dump(training_set, tr)

    else:
        with open(training_set_location, 'rb') as tr:
            training_set = pickle.load(tr)

    logger.info('Creating sentiment analysis model')
    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(
            loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
            max_iter=5, tol=None
        ))
    ])

    logger.info('Fitting test set to the model')
    clf.fit(training_set.data, training_set.labels)
    
    logger.info('Using trained model to make predictions')
    predicted = clf.predict(test_set.data)
    
    rand_num = np.random.randint(0, len(test_set.data))
    logger.info('Sample prediction: %s: %s\n Actual label: %s', test_set.data[rand_num], test_set.label_names[predicted[rand_num]], test_set.label_names[test_set.labels[rand_num]])
    logger.info('Model accuracy: %s', np.mean(predicted == test_set.labels))
    logger.info('\n%s', metrics.classification_report(test_set.labels, predicted, target_names=test_set.label_names, zero_division=0))    

    wrong_labels = np.array(predicted != test_set.labels)
    logger.info(wrong_labels[:5])