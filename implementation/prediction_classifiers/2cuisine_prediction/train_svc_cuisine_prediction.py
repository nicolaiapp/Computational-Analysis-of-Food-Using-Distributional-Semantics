from __future__ import division
import os

os.environ['SKLEARN_SITE_JOBLIB'] = '1'  # to avoid multithreading issues

import argparse
from collections import Counter
from datetime import datetime
from gensim.models import KeyedVectors, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib
import math
import numpy as np
import pandas as pd
import time

print('starting...:')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':  # to avoid multithreading issues

    def merge_cuisines(ing_list2cuisine):  # deClerq 19
        western = [['EasternEuropean', 'NorthernEuropean', 'WesternEuropean', 'NorthAmerican'], 'Western']
        eastern = [['EastAsian', 'SoutheastAsian'], 'Eastern']
        south_asian = [['SouthAsian'], 'SouthAsian']
        southern = [['African', 'LatinAmerican', 'MiddleEastern', 'SouthernEuropean'], 'Southern']

        merged_cuisine_ing_list2cuisine = ing_list2cuisine.copy()
        for ing_list in merged_cuisine_ing_list2cuisine:
            if ing_list[1] in western[0]:
                ing_list[1] = western[1]
            elif ing_list[1] in eastern[0]:
                ing_list[1] = eastern[1]
            elif ing_list[1] in south_asian[0]:
                ing_list[1] = south_asian[1]
            elif ing_list[1] in southern[0]:
                ing_list[1] = southern[1]
        return merged_cuisine_ing_list2cuisine


    def mk_joint_vocab_ing_list2cuisine(model, ing_list2cuisine):
        joint_vocab_ing_list2cuisine = []
        for ing_list_tuple in ing_list2cuisine:
            ing_list_to_append = []
            for ing in ing_list_tuple[0]:
                if ing in model:
                    ing_list_to_append.append(ing)
            if ing_list_to_append:
                ing_list2cuisine_tuple_to_append = (ing_list_to_append, ing_list_tuple[1])
                joint_vocab_ing_list2cuisine.append(ing_list2cuisine_tuple_to_append)

        print('Number of ing_lists usable with this model: ', len(joint_vocab_ing_list2cuisine))
        return joint_vocab_ing_list2cuisine


    def mk_cuisine_labels(ing_list2cuisine):
        cuisine_labels = list(set([ing_list_tuple[1] for ing_list_tuple in ing_list2cuisine]))
        cuisine_labels.sort()
        print('Number of cuisines: ', len(cuisine_labels))

        cuisine_counts = [cuisine_labels.index(ing_list_tuple[1]) for ing_list_tuple in ing_list2cuisine]
        for i, cuisine in enumerate(cuisine_labels):
            print(cuisine_counts.count(i), ' ', cuisine_labels[i], ' ing_lists')
        return cuisine_labels


    def mk_Y(joint_vocab_ing_list2cuisine):
        cuisine_labels = mk_cuisine_labels(joint_vocab_ing_list2cuisine)
        return np.array([cuisine_labels.index(ing_list_tuple[1])
                         for ing_list_tuple in joint_vocab_ing_list2cuisine])


    def mk_joint_vocab_ing_lists_with_only_one_ing(joint_vocab_ing_list2cuisine):
        return [ing_list_tuple for ing_list_tuple in joint_vocab_ing_list2cuisine if len(ing_list_tuple[0]) == 1]


    def mk_joint_vocab_ing_lists_with_only_one_ing_for_cuisine(joint_vocab_ing_list2cuisine,
                                                               cuisine_string='EastAsian'):
        return [ing_list_tuple for ing_list_tuple in joint_vocab_ing_list2cuisine
                if len(ing_list_tuple[0]) == 1
                if ing_list_tuple[1] == cuisine_string]


    # creating representations
    # 0.  Representation
    def mk_ing_list2simple_avged_model_vec(model, joint_vocab_ing_list2cuisine):
        joint_vocab_ing_lists = [ing_list_tuple[0] for ing_list_tuple in joint_vocab_ing_list2cuisine]
        ing_list2simple_avged_model_vec = [np.sum([model[ing] for ing in ing_list], axis=0)
                                           for ing_list in joint_vocab_ing_lists]
        return np.array(ing_list2simple_avged_model_vec)


    # 3. Representation
    def mk_ing_list2tfidf_vec_fitting_tfidf_vectorizer(joint_vocab_ing_list2cuisine_cvtrain,
                                                       joint_vocab_ing_list2cuisine_test):
        ing_list2cuisine = np.append(joint_vocab_ing_list2cuisine_cvtrain, joint_vocab_ing_list2cuisine_test, axis=0)
        tfidf_vectorizer = TfidfVectorizer(preprocessor=lambda x: x,
                                           tokenizer=lambda x: x,
                                           norm='l1',
                                           use_idf=True)  # already preprc+tknzd
        tfidf_vectorizer.fit_transform([ing_list[0] for ing_list in ing_list2cuisine])
        ing_list2tfidf_vec_cvtrain = tfidf_vectorizer.transform(
            [ing_list[0] for ing_list in joint_vocab_ing_list2cuisine_cvtrain])
        ing_list2tfidf_vec_test = tfidf_vectorizer.transform(
            [ing_list[0] for ing_list in joint_vocab_ing_list2cuisine_test])
        return ing_list2tfidf_vec_cvtrain, ing_list2tfidf_vec_test, tfidf_vectorizer


    def mk_ing_list_representations_and_shape_dict(model,
                                                   joint_vocab_ing_list2cuisine_cvtrain,
                                                   joint_vocab_ing_list2cuisine_test):

        # 3 cvtrain+test
        ing_list2tfidf_vec_cvtrain, ing_list2tfidf_vec_test, tfidf_vectorizer = \
            mk_ing_list2tfidf_vec_fitting_tfidf_vectorizer(joint_vocab_ing_list2cuisine_cvtrain,
                                                           joint_vocab_ing_list2cuisine_test)

        # 0
        ing_list2simple_avged_model_vec_cvtrain = mk_ing_list2simple_avged_model_vec(
            model, joint_vocab_ing_list2cuisine_cvtrain)
        ing_list2simple_avged_model_vec_test = mk_ing_list2simple_avged_model_vec(
            model, joint_vocab_ing_list2cuisine_test)

        # 1
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        ing_list2tfidf_avged_model_vec_cvtrain = ing_list2tfidf_vec_cvtrain.dot(
            np.array([model[ing] for ing in tfidf_feature_names]))
        ing_list2tfidf_avged_model_vec_test = ing_list2tfidf_vec_test.dot(
            np.array([model[ing] for ing in tfidf_feature_names]))

        # 2
        ing_list2tfidf_avged_model_vec_concat_2tfidf_vec_cvtrain = np.concatenate(
            [ing_list2tfidf_vec_cvtrain.toarray(), ing_list2tfidf_avged_model_vec_cvtrain], axis=1)
        ing_list2tfidf_avged_model_vec_concat_2tfidf_vec_test = np.concatenate(
            [ing_list2tfidf_vec_test.toarray(), ing_list2tfidf_avged_model_vec_test], axis=1)

        representations_dict = {
            '0_ing_lists2simple_avged_model_vec': [ing_list2simple_avged_model_vec_cvtrain,
                                                   ing_list2simple_avged_model_vec_test],
            '1_ing_list2tfidf_avged_model_vec': [ing_list2tfidf_avged_model_vec_cvtrain,
                                                 ing_list2tfidf_avged_model_vec_test],
            '2_ing_list2tfidf_avged_model_vec_concat_2tfidfvec': [
                ing_list2tfidf_avged_model_vec_concat_2tfidf_vec_cvtrain,
                ing_list2tfidf_avged_model_vec_concat_2tfidf_vec_test],
            '3ing_list2tfidf_vec': [ing_list2tfidf_vec_cvtrain, ing_list2tfidf_vec_test]
        }
        shape_dict = {
            '0_ing_lists2simple_avged_model_vec_shape': [ing_list2simple_avged_model_vec_cvtrain.shape,
                                                         ing_list2simple_avged_model_vec_test.shape],
            '1_ing_list2tfidf_avged_model_vec': [ing_list2tfidf_avged_model_vec_cvtrain.shape,
                                                 ing_list2tfidf_avged_model_vec_test.shape],
            '2_ing_list2tfidf_avged_model_vec_concat_2tfidfvec': [
                ing_list2tfidf_avged_model_vec_concat_2tfidf_vec_cvtrain.shape,
                ing_list2tfidf_avged_model_vec_concat_2tfidf_vec_test.shape],
            '3_ing_list2tfidf_vec': [ing_list2tfidf_vec_cvtrain.shape, ing_list2tfidf_vec_test.shape]
        }
        return representations_dict, shape_dict


    def my_train_test_stratified_split(x, y, test_size=0.1):
        skf = StratifiedKFold(n_splits=int(1 / test_size), random_state=0)
        for train_index_list, test_index_list in skf.split(x, y):
            x_train, x_test = x[train_index_list], x[test_index_list]
            y_train, y_test = y[train_index_list], y[test_index_list]
        print('len(X_train), len(X_test): ', x_train.shape[0], ', ', x_test.shape[0])

        print('Counter(list(y_train)', Counter(list(y_train)))
        print('Counter(list(y_test) ', Counter(list(y_test)))

        return x_train, x_test, y_train, y_test


    logregstring = 'logreg'
    svcstring = 'svc'
    cuisine_prediction_name = 'cuisine_prediction_' + svcstring + '__'
    cuisine_prediction_name = 'cuisine_prediction_' + logregstring + '__'


    def my_train_and_evaluate(model_name,
                              representation_name,
                              cvtrain_and_test_representation,
                              ing_list2cuisine_cvtrain_,
                              ing_list2cuisine_test_):
        start_time = time.time()
        file_name_base = cuisine_prediction_name + model_name + '_' + representation_name + '__'
        print('-------------------STARTING TRAINING FOR ' + file_name_base + '-------------------')
        print('-------------------STARTING TRAINING FOR ' + file_name_base + '-------------------')
        print('-------------------STARTING TRAINING FOR ' + file_name_base + '-------------------')
        print('Getting data for model: ', model_name, ' and ing_list representation: ', representation_name)
        # X_train, X_test, Y_train, Y_test = my_train_test_stratified_split(X, Y)
        X_train, X_test, Y_train, Y_test = cvtrain_and_test_representation[0], \
                                           cvtrain_and_test_representation[1], \
                                           ing_list2cuisine_cvtrain_, \
                                           ing_list2cuisine_test_
        macro_f1_scorer = make_scorer(f1_score, average='macro')
        parameters = [{
            # 'C': [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001],
            'C': [500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.01, 0.05],
            # 'C': [0.1, 0.5, 1, 2, 5, 10, 50, 100],
            'max_iter': [1000],  # Default: LinearSVC = 1000, LogisticRegression = 100
            # max_iter': [1],
            # 'C': [1],
            # 'multi_class': ['ovr', 'crammer_singer'],
            # 'loss': ['hinge', 'squared_hinge'],  # Default LinearSVC: squared_hinge
            # 'tol': [1e-06, 1e-04],
            # 'penalty': ['l1', 'l2'],  # Default: LinearSVC, logReg = 'l2', better change to l1; see also solver
            # 'class_weight': ['balanced', None],
            'class_weight': ['balanced'],  # Default: LinearSVC = None, better change to balanced if only 1 param
            # 'multi_class': ['auto'],  # Default: LinearSVC = 'ovr', better make explicit bec of future changes
            # 'solver': ['liblinear', 'sag'],  # liblinear, saga = l1, sag = l2; l2 potentially better against overfit
            # 'solver': ['liblinear'],  # Default: LinearSVC = 'liblinear', better make explicit because future changes
            # 'solver': ['liblinear'],  # Default: logReg = 'liblinear', better make explicit because future changes
            'dual': [False, True],
            'random_state': [0]
        }]

        # gridsearch_clf = GridSearchCV(LogisticRegression(),
        gridsearch_clf = GridSearchCV(LinearSVC(),
                                      parameters,
                                      cv=10,
                                      scoring=macro_f1_scorer,
                                      return_train_score=True,
                                      iid=False,
                                      error_score=np.nan,  # to avoid interrupt if cv zero using LogisticRegression
                                      n_jobs=-1
                                      # n_jobs=15
                                      )
        gridsearch_clf.fit(X_train, Y_train)
        best_estimator = gridsearch_clf.best_estimator_

        print('')
        print('Best parameters of model: ', model_name, ' and ', representation_name)
        print(gridsearch_clf.best_params_)
        print('')

        best_params_dataframe = pd.Series(gridsearch_clf.best_params_).to_frame()
        cv_results_dataframe = pd.DataFrame(gridsearch_clf.cv_results_)

        Y_cv_predicted = cross_val_predict(best_estimator, X_train, Y_train, cv=10)
        Y_test_predicted = best_estimator.predict(X_test)
        list_of_vars = [X_train, X_test, Y_train, Y_test, Y_cv_predicted, Y_test_predicted]
        joblib.dump(list_of_vars, file_name_base + 'list_of_vars.joblib.pkl', compress=9)  # save variables

        cvtrain_report = classification_report(Y_train, Y_cv_predicted, output_dict=True)
        test_report = classification_report(Y_test, Y_test_predicted, output_dict=True)
        cvtrain_report_df = pd.DataFrame(cvtrain_report).transpose()
        test_report_df = pd.DataFrame(test_report).transpose()
        cvtrain_report_df = cvtrain_report_df[
            ['precision', 'recall', 'f1-score', 'support']]  # == reordering (<-- bug?)
        test_report_df = test_report_df[['precision', 'recall', 'f1-score', 'support']]  # == reordering (<-- bug?)

        joblib.dump(gridsearch_clf, file_name_base + 'gridsearch_clf.joblib.pkl', compress=9)  # save model
        cv_results_dataframe.to_pickle(file_name_base + 'cv_results_df.pkl')
        best_params_dataframe.to_pickle(file_name_base + 'best_params_df.pkl')
        cvtrain_report_df.to_pickle(file_name_base + 'classification_report_cvtrain_df.pkl')
        test_report_df.to_pickle(file_name_base + 'classification_report_test_df.pkl')

        end_time = time.time()
        training_duration = (end_time - start_time) / 60
        training_duration_rounded = int(math.ceil(training_duration))
        print('-------------------DONE WITH ' + file_name_base + '-------------------')
        print('-------------------training_duration was: ' + str(
            training_duration_rounded) + ' minutes-------------------')
        print('-------------------DONE WITH ' + file_name_base + '-------------------')
        print('')
        print('')
        print('')
        print('')
        print('')
        return training_duration_rounded


    def run(model_dict, ing_list2cuisine_cvtrain_, ing_list2cuisine_test_):
        start_time = time.time()
        print('Now beginning training this many models: ', len(model_dict))
        for model_name, model in model_dict.items():
            training_duration_dict = {}
            joint_vocab_ing_list2cuisine_cvtrain = mk_joint_vocab_ing_list2cuisine(model, ing_list2cuisine_cvtrain_)
            joint_vocab_ing_list2cuisine_test = mk_joint_vocab_ing_list2cuisine(model, ing_list2cuisine_test_)
            representations_dict, shape_dict = mk_ing_list_representations_and_shape_dict(
                model,
                joint_vocab_ing_list2cuisine_cvtrain,
                joint_vocab_ing_list2cuisine_test)

            pd.Series(shape_dict).to_frame().to_pickle(
                cuisine_prediction_name + model_name + '_shape_dict_df.pkl')

            Y_cvtrain = mk_Y(joint_vocab_ing_list2cuisine_cvtrain)
            Y_test = mk_Y(joint_vocab_ing_list2cuisine_test)
            for representation_name, cvtrain_and_test_representation in representations_dict.items():
                training_duration = my_train_and_evaluate(model_name,
                                                          representation_name,
                                                          cvtrain_and_test_representation,
                                                          Y_cvtrain,
                                                          Y_test)
                training_duration_dict[representation_name] = training_duration
            pd.Series(training_duration_dict).to_frame().to_pickle(
                cuisine_prediction_name + model_name + '_train_duration_dict_df.pkl')

        end_time = time.time()
        run_duration = (end_time - start_time) / 60
        run_duration_rounded = int(math.ceil(run_duration))
        print(
            'It took ', run_duration_rounded, ' minutes to train this many model-representation-combinations: ',
            len(model_dict) * len(representations_dict))

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE ')
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


    # ##########################################################################################################
    # LOAD DATA, MODELS AND RUN THE TRAINING
    # ##########################################################################################################

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
    path_to_ahn_ing_list2cuisine_cvtrain = data_path + 'ahn_flavour_network/ing_list2cuisine_cvtrain.pkl.npy'
    path_to_ahn_ing_list2cuisine_test = data_path + 'ahn_flavour_network/ing_list2cuisine_test.pkl.npy'
    path_to_ahn_merged_ing_list2cuisine_cvtrain = '../cvtrain_and_test_data/merged_ing_list2cuisine_cvtrain.pkl.npy'
    path_to_ahn_merged_ing_list2cuisine_test = '../cvtrain_and_test_data/merged_ing_list2cuisine_test.pkl.npy'

    parser = argparse.ArgumentParser(description='cuisine prediction parameters')
    parser.add_argument('--model', default='model_im2rec_base', type=str)
    opts = parser.parse_args()

    if opts.model == 'model_googlenews':
        models = {
            'model_googlenews': KeyedVectors.load_word2vec_format(path_to_model_googlenews, binary=True),
        }

    if opts.model == 'model_wiki_fasttext':
        models = {
            'model_wiki_fasttext': FastText.load_fasttext_format(path_to_model_wiki_fasttext_bin).wv,
        }

    if opts.model == 'model_im2rec_joint_null':
        models = {
            'model_im2rec_joint_null': KeyedVectors.load_word2vec_format(path_to_model_im2rec_joint_null, binary=False),
        }

    if opts.model == 'model_im2rec_joint_avg':
        models = {
            'model_im2rec_joint_avg': KeyedVectors.load_word2vec_format(path_to_model_im2rec_joint_avg, binary=False),
        }

    if opts.model == 'model_im2rec_base':
        models = {
            'model_im2rec_base': KeyedVectors.load_word2vec_format(path_to_model_im2rec_base, binary=True),
        }

    if opts.model == 'model_im2rec_fasttext':
        models = {
            'model_im2rec_fasttext': FastText.load_fasttext_format(path_to_model_im2rec_fasttext).wv
        }

    # ing_list2cuisine_cvtrain = np.load(path_to_ahn_ing_list2cuisine_cvtrain)
    # ing_list2cuisine_test = np.load(path_to_ahn_ing_list2cuisine_test)
    ing_list2cuisine_cvtrain = np.load(path_to_ahn_merged_ing_list2cuisine_cvtrain)
    ing_list2cuisine_test = np.load(path_to_ahn_merged_ing_list2cuisine_test)

    run(models, ing_list2cuisine_cvtrain, ing_list2cuisine_test)
    print('ended...:')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
