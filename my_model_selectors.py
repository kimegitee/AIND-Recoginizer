import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_score = float('inf')
        best_model = None
        # Regularizer term, favors simpler models
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                # Number of free parameters:
                # n*(n-1) transition probabilities
                # n-1 starting probabilities
                # n*n_features means
                # n*n_features covariances
                logL = model.score(self.X, self.lengths)
                n_samples, n_features = self.X.shape
                n_params = n*n + 2*n*n_features - 1
                logN = math.log(n_samples)
                score = -2*logL + n_params*logN 
                # Smaller BIC is better
                if score < best_score:
                    best_score = score
                    best_model = model
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float('-inf')
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL_self = model.score(self.X, self.lengths)
                other_Xlengths = {k: v for k, v in self.hwords.items() if k is not self.this_word}
                logL_other = sum(model.score(X, lengths)
                    for X, lengths in other_Xlengths.values()) / len(other_Xlengths)
                score = logL_self - logL_other
                # Bigger DIC is better
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        best_score = float('-inf')
        best_n_components = self.min_n_components

        if len(self.sequences) <= 2:
            # Delegate to DIC if there are too few training examples
            return SelectorDIC(
                self.words,
                self.hwords,
                self.this_word,
                self.n_constant,
                self.min_n_components,
                self.max_n_components,
                self.random_state
            ).select()

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                logL_sum = 0
                n_splits = 3
                for train_idx, test_idx in KFold(n_splits=n_splits).split(self.sequences):
                    train_X, train_lengths = combine_sequences(train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(test_idx, self.sequences)
                    # Not using base_model to avoid redundant fitting
                    model = GaussianHMM(
                        n_components=n,
                        covariance_type="diag",
                        n_iter=1000,
                        random_state=self.random_state,
                        verbose=False
                    ).fit(train_X, train_lengths)
                    logL_sum += model.score(test_X, test_lengths)
                score = logL_sum / n_splits
                if score > best_score:
                    best_score = score
                    best_n_components = n
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
                continue

        return self.base_model(best_n_components)
