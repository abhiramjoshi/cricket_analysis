import logging
import pickle
import sys
from matplotlib.pyplot import vlines
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier
import codebase.match_data as match
import codebase.settings as settings
import pandas as pd
import numpy as np
import os
import codebase.analysis_functions as af
import utils
import codebase.web_scrape_functions as wsf
import numpy as np
from codebase.match_data import MatchData
from utils import GridSearchCV_Self_Implemented, logger
from codebase.settings import DATA_LOCATION, ANALYSIS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.fixes import loguniform

years = ['2017', '2021']
LABEL_DATA_FILENAME= f"train_commentary_labels_{years[0]}_{years[-1]}.json"
FORCE_LABEL = False
FORCE_BUNCH = True
SHUFFLE = True
TEST_TRAINING_SPLIT = 0.9
SHRINK_DATA = None
MODEL_DATA = True
FIT_METHOD = 'gs'
FIT_METHODS = {
    'single': 0,
    'gs': 1,
    'cv': 2,
    'random': 3
}
SEARCH_PARAMETERS = {
   #'vect__ngram_range': [(1, 1), (1, 2)],
   #'tfidf__use_idf': [True, False],
   #'clf__kernel': ['linear', 'poly', 'rbf'],
   'clf__alpha': [1e-4,1e-3,1e-2,1e-1],
   'clf__eta0': [0.5,1,10],
   # 'clf__fit_intercept': [True, False],
   #'clf__class_weight':['balanced'],
   'clf__verbose':[True]
}
SAVE_MODEL = False
MODEL = Perceptron
MODEL_PARAMETERS ={
    #'C':0.5,
    #'class_weight':'balanced',
    #'kernel':'linear',
    #'verbose':True
    # 'n_components': 8
    # 'n_jobs': -1
}
METRIC = geometric_mean_score
SCORING = {
    'G-Mean':metrics.make_scorer(METRIC, average='macro')
}
MODEL_DIR = os.path.join(ANALYSIS, 'model_performance_tests')

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

LOGNAME = 'Perceptron_GridSearch_Model'
if LOGNAME != '':
    log_handler = logging.FileHandler(os.path.join(MODEL_DIR, f'{LOGNAME}.log'))
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_handler)

if __name__ == "__main__":
    try:
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
                    af.create_dummies(comms, value_mapping={
                        'isOne':[1],
                        'isTwo':[2],
                        'isThree':[3],
                        'isFive':[5]
                    })
                    comm_w_labels = af.create_labels(comms, ['isWicket', 'isFour', 'isSix', 'isOne', 'isTwo', 'isThree', 'isFive'], null_category='isDot')
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
                all_comms = all_comms.sample(frac=1, random_state=42)
            
            if SHRINK_DATA is not None:
                all_comms = all_comms.sample(frac=1/SHRINK_DATA, random_state=42)
                logger.info('Dataset shrunk by factor of %s', SHRINK_DATA)

            n = all_comms.shape[0]
            data = all_comms['commentTextItems']
            labels = all_comms['labels']
            logger.info('\n %s', all_comms.groupby('labels').count()/all_comms.shape[0])
            logger.info('Creating training set with %s of the data', TEST_TRAINING_SPLIT)
            split = int(n*TEST_TRAINING_SPLIT)
            training_data, test_data, training_labels, test_labels = train_test_split(
                data, labels, train_size=TEST_TRAINING_SPLIT,shuffle=SHUFFLE, random_state=42)
            training_set = af.package_data(data=data[:split], labels=labels[:split])
            # training_set = af.package_data(data=training_data, labels=training_labels)
            logger.info('Training Set Distribution \n %s\nSet size: %s', af.describe_data_set(training_set, 'commentaryLabels').to_string(), len(training_set.data))
            logger.info('Creating test set')
            test_set = af.package_data(data=data[split:], labels=labels[split:], label_names=training_set.label_names)
            # test_set = af.package_data(data=test_data, labels=test_labels, label_names=training_set.label_names)
            logger.info('Test Set Distribution \n %s\nSet size:%s', af.describe_data_set(test_set, 'commentaryLabels').to_string(), len(test_set.data))
        # pprint(training_set.data[-10:])
        # pprint(training_set.labels[-10:])
        # p
        # print(training_set.label_names)

            with open(training_set_location, 'wb') as tr:
                pickle.dump(training_set, tr)

        else:
            with open(training_set_location, 'rb') as tr:
                training_set = pickle.load(tr)

        if MODEL_DATA:
            logger.info('Creating sentiment analysis model')
            logger.info('Using model %s', MODEL)
            clf = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MODEL(**MODEL_PARAMETERS))
                # ('clf', utils.RandomWeightedPredictor())
                # ('clf', utils.RandomPredictor())
                # ('clf', MultinomialNB())
                # ('clf', SGDClassifier()),
                # ('clf', Perceptron()), #Best
                # ('clf', RidgeClassifier())
            ])

            logger.info('Fitting test set to the model')
            
            if FIT_METHODS[FIT_METHOD] == 1:
                logger.info('Performing grid search cross validation')
                cv = StratifiedKFold(n_splits=5, shuffle=True)
                clf = GridSearchCV(clf, SEARCH_PARAMETERS, scoring = SCORING[list(SCORING.keys())[0]], cv=cv, n_jobs=-1, verbose=10, error_score=0)
                clf = clf.fit(training_set.data, training_set.labels)
                logger.info('Grid Score Results')
                logger.info('------------------------')
                logger.info('Best Grid search score: %s', clf.best_score_)
                for param_name in sorted(SEARCH_PARAMETERS.keys()):
                    logger.info("%s: %r", param_name, clf.best_params_[param_name])
                logger.info('Best Estimator: %s', clf.best_estimator_)
                grid_result_df = pd.DataFrame(clf.cv_results_)
                logger.info("Grid search full results \n: \t %s", grid_result_df.to_string().replace('\n', "\n\t"))
            
            elif FIT_METHODS[FIT_METHOD] == 2:
                logger.info('Performing standard cross validation')
                clf = utils.StandardCrossValidator(clf, scoring=SCORING[list(SCORING.keys())[0]], n_jobs=-1, verbose=1)
                clf = clf.fit(training_set.data, training_set.labels)
            
            elif FIT_METHODS[FIT_METHOD] == 3:
                logger.info('Performing randomized search cross validation')
                cv = StratifiedKFold(n_splits=5, shuffle=True)
                clf = RandomizedSearchCV(clf, SEARCH_PARAMETERS, scoring = SCORING[list(SCORING.keys())[0]], cv=cv, n_jobs=-1, verbose=10, error_score=0, n_iter=4)
                clf = clf.fit(training_set.data, training_set.labels)
                logger.info('Randomized Search Score Results')
                logger.info('------------------------')
                logger.info('Best Randomied search score: %s', clf.best_score_)
                for param_name in sorted(SEARCH_PARAMETERS.keys()):
                    logger.info("%s: %r", param_name, clf.best_params_[param_name])
                logger.info('Best Estimator: %s', clf.best_estimator_)
                grid_result_df = pd.DataFrame(clf.cv_results_)
                logger.info("Grid search full results \n: \t %s", grid_result_df.to_string().replace('\n', "\n\t"))
            
            else:
                try:
                    clf.fit(training_set.data, training_set.labels)
                except TypeError:
                    clf.steps.insert(2, ['dense_transform', utils.DenseTransformer()])
                    clf.fit(training_set.data, training_set.labels)
            
            logger.info('Using trained model to make predictions')
            predicted = clf.predict(test_set.data)
            # score = SCORING[list(SCORING.keys())[0]](clf.best_estimator_, test_set.data, test_set.labels)
            rand_num = np.random.randint(0, len(test_set.data))
            logger.info('Sample prediction: %s: %s\n Actual label: %s', test_set.data[rand_num], test_set.label_names[predicted[rand_num]], test_set.label_names[test_set.labels[rand_num]])
            logger.info(f'Direct w Predictions: Model {list(SCORING.keys())[0]}: %s', METRIC(test_set.labels, predicted, average='macro'))
            # logger.info(f'Make Score: Model {list(SCORING.keys())[0]}: %s', score)
            logger.info('\n%s', metrics.classification_report(test_set.labels, predicted, target_names=test_set.label_names, zero_division=0))    

            wrong_labels = np.array(predicted != test_set.labels)
            wrong_labels_examples = [f'{test_set.data[i]}, Predicted: {test_set.label_names[predicted[i]]}, Correct: {test_set.label_names[test_set.labels[i]]}' for i,v in enumerate(wrong_labels) if v]
            logger.info('Some Wrong labels were \n %s', wrong_labels_examples[:5])
            if SAVE_MODEL:
                with open(os.path.join(ANALYSIS, f'{LOGNAME}_model.p'), 'wb') as model:
                    pickle.dump(clf, model)
        else:
            logger.info('Skipping model creation')
    
    except Exception as e:
        logger.error(e, exc_info=True)