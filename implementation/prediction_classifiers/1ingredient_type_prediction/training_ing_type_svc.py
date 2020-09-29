from __future__ import division
from gensim.models import KeyedVectors, FastText
from sklearn.externals import joblib
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report, \
    precision_recall_fscore_support, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC
import math
import pandas as pd
import time
import warnings
import numpy as np
from collections import OrderedDict


# the the current original sklearn classification report doesn't compute all averages and there is no parameter about it
## see github issue --> https://github.com/scikit-learn/scikit-learn/issues/4558
## custom implementations with avgs --> https://github.com/apacha/MusicSymbolClassifier/blob/master/ModelTrainer/reporting/sklearn_reporting.py
### to have dictOutput like the current original sklearn implementations --> https://github.com/scikit-learn/scikit-learn/blob/55bf5d9/sklearn/metrics/classification.py#L1448
def my_classification_report(y_true, y_pred, labels=None, target_names=None,
                             sample_weight=None, digits=2, average='weighted', output_dict=False):
    """Build a text report showing the main classification metrics
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    digits : int
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.
    output_dict : bool (default = False)
        If True, return output as dict
    average : string, ['weighted' (default), 'binary', 'micro', 'macro']
        Determines the type of averaging performed on the data, after reporting the individual results per class:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class, including averages across classes.
        Unless specified otherwise, the reported averages are a prevalence-weighted macro-average across
        classes (equivalent to :func:`precision_recall_fscore_support` with ``average='weighted'``).
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                  precision    recall  f1-score   support
    <BLANKLINE>
         class 0       0.50      1.00      0.67         1
         class 1       0.00      0.00      0.00         1
         class 2       1.00      0.67      0.80         3
    <BLANKLINE>
    weighted avg       0.70      0.60      0.61         5
    <BLANKLINE>
    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        warnings.warn(
            "labels size, {0}, does not match size of target_names, {1}"
                .format(len(labels), len(target_names))
        )

    average_options = ('micro', 'macro', 'weighted', 'binary', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of ' + str(average_options))

    last_line_heading = average + ' avg'

    if target_names is None:
        target_names = [u'%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    rows = zip(target_names, p, r, f1, s)

    if output_dict:
        report_dict = OrderedDict([(label[0], label[1:]) for label in rows])
        for label, scores in report_dict.items():
            report_dict[label] = OrderedDict(zip(headers, [i.item() for i in scores]))
        for average in ('micro', 'macro', 'weighted'):
            line_heading = average + ' avg'
            avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=average, sample_weight=sample_weight)
            avg = [avg_p, avg_r, avg_f1, np.sum(s)]
            report_dict[line_heading] = OrderedDict(zip(headers, [i.item() for i in avg]))
        return report_dict
    else:
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)

        report += u'\n'

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, unused_s = precision_recall_fscore_support(y_true, y_pred,
                                                                         labels=labels,
                                                                         average=average,
                                                                         sample_weight=sample_weight)

        report += row_fmt.format(last_line_heading,
                                 avg_p,
                                 avg_r,
                                 avg_f1,
                                 np.sum(s),
                                 width=width, digits=digits)

        return report


# end of my_classification_report() #---------------------------------------------------------------------------------#

data_path = '../../../data/'
path_to_model_googlenews = data_path + 'w2v_googlenews_300/GoogleNews-vectors-negative300.bin'
path_to_model_wiki_fasttext_vec = data_path + 'w2v_fasttext_wiki_en/wiki.en/wiki.en.vec'
path_to_model_wiki_fasttext_bin = data_path + 'w2v_fasttext_wiki_en/wiki.en/wiki.en.bin'
path_to_model_im2rec_base = data_path + 'recipe1M/vocab.bin'
path_to_model_im2rec_fasttext = data_path + 'w2v_fasttext_im2r/w2v_fasttext_im2r_300.bin'
path_to_model_im2rec_joint_null = data_path + 'computed_im2r_vectors/ingredient_embeds.csv'
path_to_model_im2rec_joint_avg = data_path + 'computed_im2r_vectors/ingredient_embeds_w_avg_instr_vec.csv'
path_to_ahn_recipes_with_cuisines = data_path + 'ahn_flavour_network/srep00196-s3.csv'
path_to_ahn_ingr_info = data_path + 'ahn_flavour_network/ingr_info.tsv'
path_to_ahn_comp_info = data_path + 'ahn_flavour_network/comp_info.tsv'
path_to_ahn_ingr_comp = data_path + 'ahn_flavour_network/ingr_comp.tsv'
path_to_ahn_ing2cat_cvtrain = '../data/ing2cat_cvtrain.txt'
path_to_ahn_ing2cat_test = '../data/ing2cat_test.txt'

model_googlenews = KeyedVectors.load_word2vec_format(path_to_model_googlenews, binary=True)
model_wiki_fasttext = FastText.load_fasttext_format(path_to_model_wiki_fasttext_bin)
model_im2rec_base = KeyedVectors.load_word2vec_format(path_to_model_im2rec_base, binary=True)
model_im2rec_fasttext = FastText.load_fasttext_format(path_to_model_im2rec_fasttext)
model_im2rec_joint_null = KeyedVectors.load_word2vec_format(
    path_to_model_im2rec_joint_null, binary=False)
model_im2rec_joint_avg = KeyedVectors.load_word2vec_format(
    path_to_model_im2rec_joint_avg, binary=False)

models = {
    'model_googlenews': model_googlenews,
    'model_wiki_fasttext': model_wiki_fasttext,
    'model_im2rec_joint_null': model_im2rec_joint_null,
    'model_im2rec_joint_avg': model_im2rec_joint_avg,
    'model_im2rec_base': model_im2rec_base,
    'model_im2rec_fasttext': model_im2rec_fasttext.wv
}


def mk_Xcvtrain_Ycvtrain_Xtest_Ytest(model, ing2cat_cvtrain, ing2cat_test, test_size=0.1):
    X_cvtrain = np.array([model[x] for x in ing2cat_cvtrain if x in model])
    Y_cvtrain = np.array([ing2cat_cvtrain[x] for x in ing2cat_cvtrain if x in model])
    X_test = np.array([model[x] for x in ing2cat_test if x in model])
    Y_test = np.array([ing2cat_test[x] for x in ing2cat_test if x in model])
    print('len(X_train), len(X_test): ', len(X_cvtrain), ', ', len(X_test))
    return X_cvtrain, X_test, Y_cvtrain, Y_test


macro_f1_scorer = make_scorer(f1_score, average='macro')

parameters = [{
    'C': [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001],
    'penalty': ['l2'],
    'dual': [False, True],
    'loss': ['squared_hinge'],
    'max_iter': [500],
    'tol': [1e-06, 1e-04],
    'class_weight': ['balanced', None],
    'multi_class': ['ovr', 'crammer_singer'],
    'random_state': [0]
}]

with open(path_to_ahn_ing2cat_cvtrain, 'r') as f:
    ing2cat_cvtrain = eval(f.read())

with open(path_to_ahn_ing2cat_test, 'r') as f:
    ing2cat_test = eval(f.read())

start_time = time.time()
print('Now beginning training this many models: ', len(models))
training_duration_dict = {}
for model_name, model in models.items():
    model_train_start_time = time.time()

    print('-----------------------------------------------------------------')
    print('Getting data for model: ', model_name)
    X_train, X_test, Y_train, Y_test = mk_Xcvtrain_Ycvtrain_Xtest_Ytest(model, ing2cat_cvtrain, ing2cat_test)

    gridsearch_clf = GridSearchCV(LinearSVC(),
                                  parameters,
                                  cv=10,
                                  scoring=macro_f1_scorer,
                                  return_train_score=True,
                                  iid=False,
                                  n_jobs=-1)
    gridsearch_clf.fit(X_train, Y_train)
    best_estimator = gridsearch_clf.best_estimator_

    print('')
    print('Best parameters of model: ', model_name)
    print(gridsearch_clf.best_params_)
    print('')

    Y_cv_predicted = cross_val_predict(best_estimator, X_train, Y_train, cv=10)
    Y_test_predicted = best_estimator.predict(X_test)

    best_params_dataframe = pd.Series(gridsearch_clf.best_params_).to_frame()
    cv_results_dataframe = pd.DataFrame(gridsearch_clf.cv_results_)
    file_name_base = 'ing_types_prediction_svc__' + model_name + '__'

    list_of_vars = [X_train, X_test, Y_train, Y_test, Y_cv_predicted, Y_test_predicted]
    joblib.dump(list_of_vars, file_name_base + 'list_of_vars.joblib.pkl', compress=9)  # save variables
    shape_dict = {model_name: [X_train.shape, X_test.shape]}
    pd.Series(shape_dict).to_frame().to_pickle(file_name_base + 'shape_dict_df.pkl')

    joblib.dump(gridsearch_clf, file_name_base + 'gridsearch_clf.joblib.pkl', compress=9)  # save model
    cv_results_dataframe.to_pickle(file_name_base + 'cv_results_df.pkl')
    best_params_dataframe.to_pickle(file_name_base + 'best_params_df.pkl')

    cvtrain_report = my_classification_report(Y_train, Y_cv_predicted, output_dict=True)
    test_report = my_classification_report(Y_test, Y_test_predicted, output_dict=True)
    cvtrain_report_df = pd.DataFrame(cvtrain_report).transpose()
    test_report_df = pd.DataFrame(test_report).transpose()
    cvtrain_report_df = cvtrain_report_df[['precision', 'recall', 'f1-score', 'support']]  # == reordering (<-- bug?)
    test_report_df = test_report_df[['precision', 'recall', 'f1-score', 'support']]  # == reordering (<-- bug?)
    cvtrain_report_df.to_pickle(file_name_base + 'classification_report_cvtrain_df.pkl')
    test_report_df.to_pickle(file_name_base + 'classification_report_test_df.pkl')

    model_train_end_time = time.time()
    training_duration = (model_train_end_time - model_train_start_time) / 60
    training_duration_rounded = int(math.ceil(training_duration))
    training_duration_dict[model_name] = training_duration_rounded
    pd.Series(training_duration_dict).to_frame().to_pickle(file_name_base + 'train_duration_dict_df.pkl')

    print('-------------------DONE WITH ' + file_name_base + '-------------------')
    print('-------------------training_duration was: ' + str(
        training_duration_rounded) + ' minutes-------------------')
    print('-------------------DONE WITH ' + file_name_base + '-------------------')
    print('')
    print('')
    print('')
    print('')
    print('')

end_time = time.time()
run_duration = (end_time - start_time) / 60
run_duration_rounded = int(math.ceil(run_duration))
print('It took ', run_duration_rounded, ' minutes to train this many models: ', len(models))

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print('DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE ')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
