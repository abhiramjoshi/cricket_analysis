import itertools
import logging
import os
import json
from datetime import datetime
from sklearn.model_selection import cross_validate, GridSearchCV
import numpy as np
from collections import Counter
import random
from sklearn.base import TransformerMixin
import sklearn.utils
import pandas as pd
from copy import deepcopy
import pickle
from codebase.settings import DATA_LOCATION
LOGDIR = './logs'

if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)

LOGFILE = os.path.join(LOGDIR, f'cricket_data_log_{datetime.now().strftime("%Y_%m_%d_%H%M")}.log')
WARNINGS_LOG = os.path.join(LOGDIR, f'cricket_data_warnings.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

f_logger = logging.FileHandler(LOGFILE)
s_logger = logging.StreamHandler()

f_logger.setLevel(logging.DEBUG)
s_logger.setLevel(logging.INFO)

f_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
s_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

f_logger.setFormatter(f_formatter)
s_logger.setFormatter(s_formatter)

logger.addHandler(f_logger)
logger.addHandler(s_logger)

class PlayerNotFoundError(Exception):
    pass

class PlayerNotPartOfMatch(Exception):
    pass

class NoMatchCommentaryError(Exception):
    pass

class RetiredHurtError(Exception):
    pass

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return np.squeeze(np.asarray(X.todense()))

class StandardCrossValidator():
    
    def __init__(self, clf, scoring=None, cv=None, n_jobs=None, verbose = 0) -> None:
        self.clf = clf
        self.scoring = scoring
        self.verbose = verbose
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.train_scores = []
        self.test_scores = []
        self.estimators = []

    def fit(self, X, y):
        result = cross_validate(self.clf, X, y, scoring=self.scoring, 
                                cv=self.cv, n_jobs=self.n_jobs, 
                                verbose=self.verbose, return_estimator=True
                                )
        try:
            self.train_scores = result['train_score']
        except KeyError:
            pass
        try:
            self.test_scores = result['test_score']
        except KeyError:
            pass
        try:
            self.estimators = result['estimator']     
        except KeyError:
            pass
        return self.estimators[max(range(len(self.test_scores)), key=self.test_scores.__getitem__)]

class CrossValidator():
    FIT_METHODS = {
        'gs': GridSearchCV,
        'cv': StandardCrossValidator
    }
    def __init__(self, fit_method) -> None:
        self.fit_method = fit_method

class RandomPredictor():
    def __init__(self) -> None:
        self.labels = None
        self.label_names = None

    def fit(self, X, y):
        self.label_names = set(y)
        logger.info('Labels will be selected at random, so there is nothing to learn')
    
    def predict(self, X):
        predicted = []
        for sample in X:
            label = np.random.randint(0, len(self.label_names))
            predicted.append(label)
        return predicted

class RandomWeightedPredictor():
    def __init__(self) -> None:
        self.labels = None
        self.label_names = None
        self.label_weights = None

    def fit(self, X, y):
        c = Counter(y)
        weights = [i/sum(c.values()) for i in c.values()]
        self.label_weights = weights
        logger.info(self.label_weights)
        self.label_names = list(c.keys())
        logger.info('Labels will be selected at random, so there is nothing to learn')
    
    def predict(self, X):
        predicted = []
        weights = [100*i for i in self.label_weights]
        logger.info(weights)
        for sample in X:
            label = random.choices(self.label_names, self.label_weights, k=1)
            predicted.append(label[0])
            # remainder = np.random.randint(0, 10001)
            # for i, weight in enumerate(weights):
            #     quotient = remainder // weight
            #     remainder -= weight
            #     if quotient == 0:
            #         #logger.info('predicting %s', i)
            #         predicted.append(i)
        return predicted

def describe_data_set(dataset, title, label_names=None):
    if isinstance(dataset, sklearn.utils.Bunch):
        labels_unmapped = [dataset.label_names[label] for label in dataset.labels]
    else:
        labels_unmapped = [label_names[label] for label in dataset]
    series = pd.Series(labels_unmapped, )
    groups = series_to_df(series, [title,'labels']).groupby('labels').count()/series.shape[0]
    return groups

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

class GridSearchCV_Self_Implemented():
    def __init__(self, estimator, parameters:dict, scoring, cv:int=5, n_jobs=-1, verbose=3, error_score=0) -> None:
        self.estimator = estimator
        self.parameters = parameters
        self.cv = cv
        self.scorer = scoring
        self.best_score_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.best_estimator_ = None
    
    def fit(self, X, y, label_names=None):
        scores = {}
        parameter_permutations = list(itertools.product(*[self.parameters[i] for i in self.parameters]))
        intervals = [i*len(X)//self.cv for i in range(1, self.cv+1)]
        intervals.insert(0,0)
        logger.info('Total fittings: %s', len(parameter_permutations))
        for parameter_perm in parameter_permutations:
            cv_scores = []
            model_params = {param:parameter_perm[i] for i, param in enumerate(self.parameters)}
            logger.info('Model parameters are %s', model_params)
            # model_params = defaultdict(dict)
            # for i, param in enumerate(self.parameters):
            #     model_params[param.partition('__')[0]][param.partition('__')[-1]] = parameter_perm[i]
            # for component in model_params:    
            #     self.estimator = self.estimator[component](**model_params)
            estimator = self.estimator.set_params(**model_params)
            for i in range(1,self.cv+1):
                logger.info('Fitting %s of %s', i, self.cv)
                if self.cv == 1:
                    train_data = X
                    train_labels = y
                else:
                    train_data = X[:intervals[i-1]] + X[intervals[i]:]
                    train_labels = np.concatenate((y[:intervals[i-1]], y[intervals[i]:]), axis=0)
                try:
                    logger.info('Training Set Distribution \n %s \nSet size: %s', describe_data_set(train_labels, 'commentaryLabels', label_names).to_string(), len(train_data))
                except TypeError:
                    pass
                val_data = X[intervals[i-1]:intervals[i]]
                val_labels = y[intervals[i-1]:intervals[i]]
                try:
                    logger.info('Validation Set Distribution \n %s\nSet size: %s', describe_data_set(val_labels, 'commentaryLabels', label_names).to_string(), len(val_data))
                except TypeError:
                    pass
                model = estimator.fit(train_data, train_labels)
                score = self.scorer(model, val_data, val_labels)
                logger.info('Validation score: %s', score)
                cv_scores.append(score)
            av = sum(cv_scores)/len(cv_scores)
            scores[parameter_perm] = av
            if av >= max(scores.values()):
                self.best_estimator_ = deepcopy(model)
                self.best_params_ = model_params
                self.best_score_ = av
        
        for score in scores:
            logger.info('Score for %s: %s', score, scores[score])

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

def load_data(match_id, suffix, data_folder=DATA_LOCATION, subfilepath='', file_ext='json'):
    try:
        if os.path.exists(os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.p')):
                with open(os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.p'), 'rb') as jf:
                    logger.info(f"Loading data from {os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.p')}")
                    return pickle.load(jf)
        elif os.path.exists(os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.{file_ext}')):
                with open(os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.{file_ext}'), 'r', encoding='utf-8') as jf:
                    logger.info(f"Loading data from {os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.{file_ext}')}")
                    if file_ext == 'json':
                        return json.load(jf)
                    else:
                        return jf.read()
    except UnicodeDecodeError:
        pass
    
    return False

def save_data(match_id, data, suffix, data_folder=DATA_LOCATION, subfilepath='', file_ext='json', serialize=True):
    if serialize:
        with open(os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.p'), 'wb') as f:
            logger.info(f"Saving data to {os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.p')}")
            pickle.dump(data, f)
    else: 
        with open(os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.{file_ext}'), 'w', encoding='utf-8') as f:
            logger.info(f"Saving data to {os.path.join(data_folder, subfilepath, f'{match_id}_{suffix}.{file_ext}')}")
            if file_ext == 'json':
                f.write(json.dumps(data))
            else:
                f.write(data)


def check_if_ipython():
    """Returns true if code running in IPython Shell"""
    try:
        shell = get_ipython().__class__.__name__
        return True
    except NameError:
        return False
