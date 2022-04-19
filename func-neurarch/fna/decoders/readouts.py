import copy

import numpy as np
from sklearn import linear_model as lm, metrics as met, svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from fna.tasks.symbolic import embeddings
from fna.tools.visualization import plotting
from fna.tools import parameters as par
from fna.tools import utils
from .output_mapper import OutputMapper

logger = utils.logger.get_logger(__name__)


class Readout(object):
    """
    Readout object, trained to produce an estimation y(t) of output by reading out population state variables.
    Each readout is associated with a single state matrix / extractor, as well as a task - subtask pair (y(t)).
    The labels of these variables are stored as member variables in the readout object.
    """
    def __init__(self, label, readout_parameters, rng=None, display=True):
        """
        Create and initialize Readout object.

        :param label: [str] Unique readout label composed of names of the task, state matrix and readout.
        :param readout_parameters: ParameterSet object specifying Readout parameters
        """
        self.initializer = self._parse_parameters(readout_parameters)

        self.label = label
        self.algorithm = readout_parameters.algorithm
        self.task = readout_parameters.task
        self.extractor = readout_parameters.extractor

        self.weights = OutputMapper(label, self.initializer)
        self.fit_obj = None
        self.output = None
        self.test_target = None
        self.norm_wout = None
        self.performance = {}

        if rng is None:
            self.rng_state = np.random.RandomState()
            logger.warning("Readout results may not be reproducible (RNG not set)!")
        else:
            # scikit-learn currently does not support the new numpy.random.default_state generator, so fall back
            self.rng_state = np.random.RandomState(seed=rng.integers(low=1, high=1e6, size=1))

        if display:
            logger.info("  - {0} trained with {1} on {2}, using state {3}".format(
                self.label, self.algorithm, self.task, self.extractor))

    @staticmethod
    def _parse_parameters(initializer):
        """
        Sanitize readout parameters and ensure required parameter entries are present.

        :param initializer:
        :return:
        """
        if isinstance(initializer, dict):
            initializer = par.ParameterSet(initializer)
        assert isinstance(initializer, par.ParameterSet), "Readout must be initialized with ParameterSet or " \
                                                          "dictionary"
        required_keys = ['algorithm', 'task', 'extractor']
        par.validate_keys(initializer, required_keys)

        # by default, we do not store intermediate readout weights (per batch)
        if 'save' not in initializer:
            initializer['save'] = False

        return initializer

    @staticmethod
    def _filter_accepted_ids(accepted, state_matrix, target):
        if accepted is not None:
            state_matrix = state_matrix[:, accepted]
            if len(target.shape) == 1:
                target = target[accepted]
            elif len(target.shape) == 2:
                target = target[:, accepted]
            else:
                raise ValueError('Maximum supported dimensionality of target matrix is 2!')
        return state_matrix, target

    def parse_targets(self, target, vocabulary=None):
        """
        Parse and pre-process target outputs to the correct numerical values
        :param target:
        :param vocabulary: [int] size of vocabulary (e.g. number of unique targets)

        :return:
        """
        # binary target is only valid if the entries are floats and 0. or 1.
        is_binary_target = len(np.unique(target)) <= 2 and np.all([x in [0., 1.] for x in np.unique(target)]) \
                           and issubclass(target.dtype.type, np.floating)
        # labeled target is only valid if the entries are integers or strings, and the array is one dimensional
        is_labeled_target = (len(target.shape) == 1 and (isinstance(target[0], str) or isinstance(target[0], int)))

        assert is_binary_target or is_labeled_target, "Could not retrieve target properties (binary/labeled?) !"
        # set binary_target and target_labels
        if not is_binary_target:
            if vocabulary is None:
                vocabulary = np.unique(target)

            tg = embeddings.VectorEmbeddings(vocabulary=vocabulary, rng=self.rng_state).one_hot()
            binary_target = tg.draw_stimulus_sequence(target, as_array=True, verbose=False)
        else:
            binary_target = target

        if not is_labeled_target:
            # before
            target_labels = np.argmax(target, 0)
            # TODO @zbarni this should be a 1D array of length target.shape[0], right?
            # if np.count_nonzero(target[0, :]) == 1:
            #     target_labels = np.argmax(target, 1)
            # else:
            #     target_labels = np.argmax(target, 0)
        else:
            target_labels = target
        return binary_target, target_labels

    def parse_outputs(self, output, method='k-WTA', k=1, vocabulary=False, verbose=False):
        """
        Parse and pre-process network outputs to the correct format
        :param output: estimated output [numpy.array (binary or real-valued) or list (labels)]
        :param method: possible values: 'k-WTA', 'threshold', 'softmax'
        :param k: [int]
            k-WTA: number of winners, applicable for the k-WTA scenario
            threshold: all larger values than this are set to one
        :param vocabulary: list of symbols/numbers mapped to the vocabulary of the sequence
        :return:

        # TODO in some cases when the method is explicitly passed as None, the returned binary_output is actually
        # TODO the raw readout output. This error can propagate downstream to the results dictionary and lead to errors
        # TODO in the 'MAE' entries. Solution: don't allow method=None !
        """
        # binary output is only valid if the entries are floats and 0. or 1.
        is_binary_output = len(np.unique(output)) <= 2 and np.all([x in [0., 1.] for x in np.unique(output)]) \
                           and issubclass(output.dtype.type, np.floating)
        # labeled output is only valid if the entries are integers or strings, and the array is one dimensional
        is_labeled_output = (len(output.shape) == 1 and
                             (isinstance(output[0], str) or np.issubdtype(output[0], np.integer)))

        if not is_labeled_output and method is not None:
            output_labels = np.argmax(output, 0)
            if method == 'k-WTA':
                binary_output = np.zeros((output.shape[0], len(output_labels)))
                for kk in range(output.shape[1]):
                    args = np.argsort(output[:, kk])[-k:]
                    binary_output[args, kk] = 1.

            elif method == 'threshold':
                if verbose:
                    logger.info("Applying a threshold to readout outputs")
                binary_output = np.zeros((output.shape[1], len(output_labels)))
                for kk in range(output.shape[1]):
                    binary_output[np.where(output[:, kk] >= k), kk] = 1.

            elif method == 'softmax':
                if verbose:
                    logger.info("Converting outputs to probability density with softmax")
                e_x = np.exp(output - np.max(output))
                binary_output = e_x / e_x.sum(axis=0)

            elif method is not None:
                raise ValueError("Unknown method for output processing! Exiting..")

            else:
                binary_output = output
        else:
            if not is_binary_output and is_labeled_output:
                tg = embeddings.VectorEmbeddings(vocabulary=vocabulary, rng=self.rng_state).one_hot()
                binary_output = tg.draw_stimulus_sequence(output, as_array=True, verbose=False)
            else:
            # TODO this may in some cases be the raw readout output, not the binarized version
                binary_output = output
            output_labels = output

        return binary_output, output_labels

    def _does_incremental_learning(self):
        """
        Verify if the readout supports incremental learning
        """
        if not hasattr(self.fit_obj, "partial_fit") and not self.algorithm == 'force' or self.algorithm == 'FORCE':
            logger.warning("Readout {} does not support incremental learning, model fitting will overwrite previous "
                           "results".format(self.label))

    def train(self, batch_label, state_matrix_train, target_train, accepted=None,
              vocabulary=None, display=True):
        """
        Train readout.

        :param batch_label:
        :param state_matrix_train: [np.ndarray] state matrix
        :param target_train: [np.ndarray]
        :param accepted:
        :param vocabulary: vocabulary
        :param display:
        :return:
        """
        assert (isinstance(state_matrix_train, np.ndarray)), "Provide state matrix as array"
        assert (isinstance(target_train, np.ndarray)), "Provide target matrix as array"

        state_matrix_train, target_train = self._filter_accepted_ids(accepted, state_matrix_train, target_train)
        if not target_train.shape[0] == 1 and not isinstance(target_train[0], float):
            target_train, _ = self.parse_targets(target_train, vocabulary)

        if display:
            logger.info("  - Training readout with [{}] on [{}] task".format(self.algorithm, self.task))

        if self.algorithm == 'pinv':
            if self.fit_obj is None:
                self.fit_obj = lm.LinearRegression(fit_intercept=False, n_jobs=-1)

            self.fit_obj.fit(state_matrix_train.T, target_train.T)
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'ridge':
            if self.fit_obj is None:
                alphas = 10.0 ** np.arange(-5, 4)
                self.fit_obj = lm.RidgeCV(alphas, fit_intercept=False)
            self.fit_obj.fit(state_matrix_train.T, target_train.T)
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'logistic':
            if self.fit_obj is None:
                C = 10.0 ** np.arange(-5, 5)
                n_folds = np.min(np.unique(target_train, axis=1, return_counts=True)[1])
                if n_folds < 2:
                    raise ValueError("Batch size too small for cross-validation on {0}".format(self.algorithm))
                logger.info("Performing {0}-fold CV for logistic regression...".format(n_folds))
                self.fit_obj = lm.LogisticRegressionCV(C, cv=n_folds, penalty='l2', dual=False, fit_intercept=False,
                                                       n_jobs=-1, multi_class='auto', max_iter=1000,
                                                       random_state=self.rng_state)
            self.fit_obj.fit(state_matrix_train.T, np.argmax(np.array(target_train), 0))
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'perceptron':
            if self.fit_obj is None:
                self.fit_obj = lm.Perceptron(fit_intercept=False, random_state=self.rng_state)

            self.fit_obj.fit(state_matrix_train.T, np.argmax(np.array(target_train), 0))
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'svm-linear':
            if self.fit_obj is None:
                self.fit_obj = svm.SVC(kernel='linear', random_state=self.rng_state)

            self.fit_obj.fit(state_matrix_train.T, np.argmax(np.array(target_train), 0))
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'svm-rbf':
            if self.fit_obj is None:
                self.fit_obj = svm.SVC(kernel='rbf', random_state=self.rng_state)
                n_folds = np.min(np.unique(target_train, axis=1, return_counts=True)[1])
                if n_folds < 2:
                    raise ValueError("Batch size too small for cross-validation on {0}".format(self.algorithm))
                logger.info("Performing {0}-fold CV for svm-rbf hyperparameters...".format(n_folds))
                # use exponentially spaces C...
                C_range = 10.0 ** np.arange(-2, 9)
                # ... and gamma
                gamma_range = 10.0 ** np.arange(-5, 4)
                param_grid = dict(gamma=gamma_range, C=C_range)
                # pick only a subset of train dataset...
                target_test = target_train
                state_test = state_matrix_train
                cv = StratifiedKFold(n_splits=n_folds)
                grid = GridSearchCV(self.fit_obj, param_grid=param_grid, cv=cv, n_jobs=-1)
                grid.fit(state_test.T, np.argmax(np.array(target_test), 0))

                # use best parameters:
                self.fit_obj = grid.best_estimator_

            self.fit_obj.fit(state_matrix_train.T, np.argmax(np.array(target_train), 0))
            self.weights.update_weights(batch_label, self.fit_obj.coef0, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        # ############# algorithms supporting batch updates ##########################
        elif self.algorithm == 'FORCE' or self.algorithm == 'force':
            output = []
            if len(target_train.shape) == 1:
                target_train = np.array([target_train])
            self.test_target = target_train

            if self.fit_obj is None:
                alpha = 1.0 # learning rate
                Nn = state_matrix_train.shape[0]
                P = np.identity(Nn) / alpha
                w_out = np.zeros((Nn, target_train.shape[0]))
                self.fit_obj = P
            else:
                w_out = self.weights.weights

            for sample in tqdm(range(target_train.shape[1]), desc="Online FORCE learning "):
                z = np.dot(w_out.T, state_matrix_train[:, sample])
                output.append(z)
                eminus = z.T - target_train[:, sample]
                denom = (1 + np.dot(state_matrix_train[:, sample].T, np.dot(self.fit_obj,
                                                                            state_matrix_train[:, sample])))
                dPdt = -1 * np.outer(np.dot(self.fit_obj, state_matrix_train[:, sample]),
                                     np.dot(state_matrix_train[:, sample].T, self.fit_obj)) / denom
                P_new = self.fit_obj + dPdt
                dwdt = -1 * np.outer(eminus, np.dot(P_new, state_matrix_train[:, sample]))
                w_out += dwdt.T

            self.output = np.array(output).T
            self.weights.update_weights(batch_label, w_out, save=self.initializer['save'])
            self.test_target = target_train

        elif self.algorithm == "sgd-reg":
            if target_train.shape[0] == 1:
                target = target_train.T.ravel()
            else:
                target = target_train.T

            if self.fit_obj is None:
                params = {
                    "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "eta0": [0.0001, 0.001, 0.01, 0.1, 1.]}
                if len(target.shape) == 1:
                    model = lm.SGDRegressor(max_iter=10000, random_state=self.rng_state)
                else:
                    model = MultiOutputRegressor(lm.SGDRegressor(max_iter=10000, random_state=self.rng_state), n_jobs=-1)
                    params = {"estimator__"+k: v for k, v in params.items()}

                clf = GridSearchCV(model, param_grid=params, n_jobs=-1)
                logger.info("Performing GridSearchCV for best SGDRegressor...")
                clf.fit(state_matrix_train.T, target)
                logger.info("Best Fit: {0} [{1}]".format(clf.best_estimator_, clf.best_score_))
                self.fit_obj = clf.best_estimator_

            self.fit_obj.partial_fit(state_matrix_train.T, target)
            if len(target.shape) == 1:
                coefs = self.fit_obj.coef_
            else:
                coefs = np.array([x.coef_ for x in self.fit_obj.estimators_])

            self.weights.update_weights(batch_label, coefs, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == "sgd-class":
            tgt = np.argmax(target_train, 0)

            if self.fit_obj is None:
                params = {
                    "loss": ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "eta0": [0.0001, 0.001, 0.01, 0.1, 1.]}
                model = lm.SGDClassifier(max_iter=100000, n_jobs=-1, random_state=self.rng_state)
                clf = GridSearchCV(model, param_grid=params, n_jobs=-1)
                logger.info("Performing GridSearchCV for best SGDClassifier...")
                clf.fit(state_matrix_train.T, tgt)
                logger.info("Best Fit: {0} [{1}]".format(clf.best_estimator_, clf.best_score_))
                self.fit_obj = clf.best_estimator_

            self.fit_obj.partial_fit(state_matrix_train.T, tgt)
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == "ridge-sgd":
            if target_train.shape[0] == 1:
                target = target_train.T.ravel()
            else:
                target = target_train.T

            if self.fit_obj is None:
                params = {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.],
                    "eta0": [0.0001, 0.001, 0.01, 0.1, 1.]
                }
                if len(target.shape) == 1:
                    model = lm.SGDRegressor(max_iter=100000, loss="squared_loss", penalty="l2", random_state=self.rng_state)
                else:
                    model = MultiOutputRegressor(
                        lm.SGDRegressor(max_iter=100000, loss="squared_loss", penalty="l2", random_state=self.rng_state),
                        n_jobs=-1)
                    params = {"estimator__"+k: v for k, v in params.items()}
                clf = GridSearchCV(model, param_grid=params, n_jobs=-1)
                logger.info("Performing GridSearchCV for Ridge-SGD...")
                clf.fit(state_matrix_train.T, target)
                logger.info("Best Fit: {0} [{1}]".format(clf.best_estimator_, clf.best_score_))
                self.fit_obj = clf.best_estimator_
            self.fit_obj.partial_fit(state_matrix_train.T, target)
            if len(target.shape) == 1:
                coefs = self.fit_obj.coef_
            else:
                coefs = np.array([x.coef_ for x in self.fit_obj.estimators_])
            self.weights.update_weights(batch_label, coefs, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == "pinv-sgd":
            if target_train.shape[0] == 1:
                target = target_train.T.ravel()
            else:
                target = target_train.T

            if self.fit_obj is None:
                params = {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.],
                    "eta0": [0.0001, 0.001, 0.01, 0.1, 1.]
                }
                if len(target.shape) == 1:
                    model = lm.SGDRegressor(loss="squared_loss", penalty="l1", max_iter=10000, random_state=self.rng_state)
                else:
                    model = MultiOutputRegressor(
                        lm.SGDRegressor(max_iter=10000, loss="squared_loss", penalty="l1", random_state=self.rng_state),
                        n_jobs=-1)
                    params = {"estimator__"+k: v for k, v in params.items()}
                clf = GridSearchCV(model, param_grid=params, n_jobs=-1)
                logger.info("Performing GridSearchCV for Linear Least Squares-SGD...")
                clf.fit(state_matrix_train.T, target)
                logger.info("Best Fit: {0} [{1}]".format(clf.best_estimator_, clf.best_score_))
                self.fit_obj = clf.best_estimator_
            self.fit_obj.partial_fit(state_matrix_train.T, target)
            if len(target.shape) == 1:
                coefs = self.fit_obj.coef_
            else:
                coefs = np.array([x.coef_ for x in self.fit_obj.estimators_])
            self.weights.update_weights(batch_label, coefs, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'logistic-sgd':
            tgt = np.argmax(target_train, 0)
            # scaler = StandardScaler() ## standardization is done at the state-matrix level
            # state_matrix_train = scaler.fit_transform(state_matrix_train)

            if self.fit_obj is None:
                params = {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "eta0": [0.0001, 0.001, 0.01, 0.1, 1.]
                }

                model = lm.SGDClassifier(max_iter=10000, loss="log", n_jobs=-1, random_state=self.rng_state)
                clf = GridSearchCV(model, param_grid=params, n_jobs=-1)
                logger.info("Performing GridSearchCV for best LogisticRegressor via SGD...")
                clf.fit(state_matrix_train.T, tgt)
                logger.info("Best Fit: {0} [{1}]".format(clf.best_estimator_, clf.best_score_))
                self.fit_obj = clf.best_estimator_

            self.fit_obj.partial_fit(state_matrix_train.T, tgt)
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train

        elif self.algorithm == 'svm-linear-sgd':
            tgt = np.argmax(target_train, 0)
            # scaler = StandardScaler()
            # state_matrix_train = scaler.fit_transform(state_matrix_train)

            if self.fit_obj is None:
                params = {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "eta0": [0.0001, 0.001, 0.01, 0.1, 1.]
                }

                model = lm.SGDClassifier(max_iter=10000, loss="hinge", n_jobs=-1, random_state=self.rng_state)
                clf = GridSearchCV(model, param_grid=params, n_jobs=-1)
                logger.info("Performing GridSearchCV for best SVM-linear Classifier via SGD...")
                clf.fit(state_matrix_train.T, tgt)
                logger.info("Best Fit: {0} [{1}]".format(clf.best_estimator_, clf.best_score_))
                self.fit_obj = clf.best_estimator_

            self.fit_obj.partial_fit(state_matrix_train.T, tgt)
            self.weights.update_weights(batch_label, self.fit_obj.coef_, save=self.initializer['save'])
            self.output = self.fit_obj.predict(state_matrix_train.T)
            self.test_target = target_train
        else:
            raise NotImplementedError("Training algorithm {} not supported.".format(self.algorithm))
        self._does_incremental_learning()
        # print(self.weights.weights)

    def predict(self, state_matrix_test, target_test, accepted=None, vocabulary=None):
        """
        Acquire readout output in test phase.

        :param state_matrix_test: [np.ndarray] state matrix
        :param target_test: [np.ndarray]
        :param accepted:
        :param vocabulary:

        :return:
        """
        logger.info("  - Testing readout with [{}] on task [{}]".format(self.algorithm, self.task))

        assert (isinstance(state_matrix_test, np.ndarray)), "Provide state matrix as array"
        assert (isinstance(target_test, np.ndarray)), "Provide target matrix as array"

        state_matrix_test, target_test = self._filter_accepted_ids(accepted, state_matrix_test, target_test)
        train_target_dims = self.test_target.shape
        if not target_test.shape[0] == 1 and not isinstance(target_test[0], float):
            self.test_target, _ = self.parse_targets(target_test, vocabulary)
            # self.test_target = self.test_target.T
        else:
            self.test_target = target_test

        # if not self.test_target.shape[0] == train_target_dims[0] or self.test_target.shape[1] == train_target_dims[1]:
        #     logger.warning("Mismatch between training and testing data dimensions!")
        # assert self.test_target.shape[0] == train_target_dims[0] or self.test_target.shape[1] == train_target_dims[1], \
        #     "Mismatch between of training and testing data dimensions!"

        if not isinstance(self.fit_obj, np.ndarray):
            self.output = self.fit_obj.predict(state_matrix_test.T)
        else:
            self.output = np.dot(self.weights.weights.T, state_matrix_test)

    def evaluate(self, process_output_method='k-WTA', symbolic=True, mse_only=False, vocabulary=None):
        """
        Compute readout performance according to different metrics.

        :param process_output_method: None (use raw output), 'k-WTA', 'threshold', 'softmax'
        :param symbolic: bool, whether the task is symbolic or analog
        :param mse_only: only compute the MSE, no label accuracy etc.
        """
        logger.info("Evaluating {}-{} performance...".format(self.label, self.task))

        if self.output is None or self.test_target is None:
            raise ValueError("Readout contains no outputs, run predict before evaluating")

        if self.output.shape != self.test_target.shape and len(self.output.shape) > 1:
            self.output = self.output.T

        if not symbolic and not self.output.shape != self.test_target.shape:
            self.performance.update({
                'raw': {
                    'MSE': met.mean_squared_error(self.test_target, self.output),
                    'MAE': met.mean_absolute_error(self.test_target, self.output),
                }})
        elif not symbolic and self.output.shape != self.test_target.shape and len(self.output.shape) == 1:
            self.performance.update({
                'raw': {
                    'MSE': met.mean_squared_error(self.test_target.ravel(), self.output),
                    'MAE': met.mean_absolute_error(self.test_target.ravel(), self.output),
                }})

        # for a symbolic task, ...
        if symbolic:
            assert vocabulary is not None, "For symbolic sequences, we need the vocabulary (size) for training/testing"
            binary_target, target_labels = self.parse_targets(self.test_target, vocabulary)

            if len(self.output.shape) == 1:  # labels
                # if the output contains labels, we need to pass an integer version of the vocabulary to create
                # a corresponding binary_output
                vocabulary_int = np.arange(len(np.unique(vocabulary)))
                binary_output, output_labels = self.parse_outputs(self.output, method=None, vocabulary=vocabulary_int)
                # k=np.unique(target_labels))  # this would lead to array dimensionality errors
                                               # if the items unique(testing) < unique(training)
            else:
                binary_output, output_labels = self.parse_outputs(self.output, method=process_output_method,
                                                                  k=len(np.unique(vocabulary)))

            if not self.output.shape != self.test_target.shape:
                self.performance.update({
                    'raw': {
                        'MSE': met.mean_squared_error(self.test_target, self.output),
                        'MAE': met.mean_absolute_error(self.test_target, self.output),
                    }})
            else:
                # print(binary_target.shape, binary_output.shape)
                self.performance.update({
                    'raw': {
                        'MSE': met.mean_squared_error(binary_target, binary_output),
                        'MAE': met.mean_absolute_error(binary_target, binary_output),
                    }})
            self.performance.update({
                'max': {
                    'MSE': met.mean_squared_error(binary_target, binary_output),
                    'MAE': met.mean_absolute_error(binary_target, binary_output),
                    'accuracy': 1 - np.mean(np.absolute(binary_target - binary_output))
                }})

            if not mse_only:
                self.performance.update({
                    'label': {
                        'accuracy': met.accuracy_score(target_labels, output_labels),
                        'hamming_loss': met.hamming_loss(target_labels, output_labels),
                        'precision': met.average_precision_score(binary_output, binary_target, average='weighted'),
                        'f1': met.f1_score(binary_target, binary_output, average='weighted'),
                        'recall': met.recall_score(target_labels, output_labels, average='weighted'),
                        'balanced_accuracy': met.balanced_accuracy_score(target_labels, output_labels, adjusted=True),
                        'kappa_score': met.cohen_kappa_score(target_labels, output_labels),
                        'mcc': met.matthews_corrcoef(target_labels, output_labels)
                    }})

                if len(self.output.shape) != 1:
                    try:
                        p_out, _ = self.parse_outputs(self.output, method='softmax')
                        self.performance['label'].update({'log_loss': met.log_loss(binary_target, p_out)})
                    except Exception:
                        self.performance['label'].update({'log_loss': np.nan})

                self.performance.update({
                    'confusion_matrix': met.confusion_matrix(target_labels, output_labels),
                    'class_support': met.precision_recall_fscore_support(target_labels, output_labels)
                })
                logger.info("\n"+ met.classification_report(target_labels, output_labels))
        return self.performance

    def copy(self):
        """
        Copy the readout object.
        :return: new Readout object
        """
        return copy.deepcopy(self)

    def reset(self):
        """
        Reset current readout.
        :return:
        """
        self.__init__(self.label, self.initializer, False)

    def plot_confusion(self, display=True, save=False):
        """
        """
        plotting.plot_confusion_matrix(self.performance['confusion_matrix'],
                                       label=self.label + self.task, display=display, save=save)


class DummyReadout(Readout):
    """
    Instantiate a dummy readout, to quantify the accuracy of internal decoders in ANNs
    """
    def __init__(self, rnn_label, output, target):
        pars = par.ParameterSet({'algorithm': 'ridge', 'task': 'representation', 'extractor': 'internal'})
        label = '{}-internal-output'.format(rnn_label)
        super().__init__(label=label, readout_parameters=pars)
        self.output = output
        self.test_target = target

