#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午8:36

import argparse
import gzip
import pickle
import numpy as np

import scipy
from data_helper import DataHelper
from eval.evaluation import Evaluator
from models.classifiers import ActionClassifier, RelationClassifier
from models.parser import RstParser


def train_model(data_helper):
    action_clf = ActionClassifier(data_helper.action_feat_template, data_helper.action_map)
    relation_clf = RelationClassifier(data_helper.relation_feat_template_level_0,
                                      data_helper.relation_feat_template_level_1,
                                      data_helper.relation_feat_template_level_2,
                                      data_helper.relation_map)
    rst_parser = RstParser(action_clf, relation_clf)
    print('extracting action feature')

    # action_fvs, action_labels, dependency_feats = list(zip(*data_helper.gen_action_train_data()))
    action_fvs, action_labels, dependency_feats = list(zip(*data_helper.gen_action_train_data()))
    # with open('..\data\model\\action\\feat.bin', 'wb') as file:
    #     pickle.dump((action_fvs, action_labels, dependency_feats), file)
    # with open('..\data\model\\action\\feat.bin', 'rb') as file:
    #     action_fvs, action_labels, dependency_feats = pickle.load(file)

    rst_parser.action_clf.train(np.concatenate([scipy.sparse.vstack(action_fvs).toarray(), dependency_feats], axis=1),
                                action_labels)

    # rst_parser.save(model_dir='..\data\model\\action')
    # rst_parser.load(model_dir='..\data\model\\action')

    # train relation classifier
    for level in [0, 1, 2]:
        relation_fvs, relation_labels, dependency_feat = list(zip(*data_helper.gen_relation_train_data(level)))
        print(np.concatenate([scipy.sparse.vstack(relation_fvs).toarray(), np.array(dependency_feat)], axis=1).shape)

        print('{} relation samples at level {}.'.format(len(relation_labels), level))
        # rst_parser.relation_clf.train(scipy.sparse.vstack(relation_fvs), relation_labels, level)
        rst_parser.relation_clf.train(
            np.concatenate([scipy.sparse.vstack(relation_fvs).toarray(), np.array(dependency_feat)], axis=1),
            relation_labels, level)

    rst_parser.save(model_dir='../../data/model/n~model')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true',
                        help='whether to extract feature templates, action maps and relation maps')
    parser.add_argument('--train', action='store_true',
                        help='whether to train new models')
    parser.add_argument('--eval', action='store_true',
                        help='whether to do evaluation')
    parser.add_argument('--pred', action='store_true',
                        help='whether to do prediction')
    parser.add_argument('--data_dir', help='train data directory')
    parser.add_argument('--eval_dir', help='eval data directory')
    parser.add_argument('--pred_dir', help='predict data directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Use brown clusters
    with gzip.open("../../data/resources/bc3200.pickle.gz") as fin:
        print('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    # print(brown_clusters)
    data_helper = DataHelper(max_action_feat_num=330000, max_relation_feat_num=300000,
                             min_action_feat_occur=1, min_relation_feat_occur=1,
                             brown_clusters=brown_clusters)
    if args.prepare:
        # Create training data
        data_helper.create_data_helper(data_dir=args.data_dir)
        data_helper.save_data_helper('../data/data_helper_average.bin')
    if args.train:
        data_helper.load_data_helper('../data/data_helper_average.bin')
        data_helper.load_train_data(data_dir=args.data_dir)
        train_model(data_helper)
    if args.eval:
        # Evaluate models on the RST-DT test set
        evaluator = Evaluator(model_dir='../../data/model/n~model')
        evaluator.eval_parser(path=args.data_dir, report=True, bcvocab=brown_clusters, draw=True)

    if args.pred:
        # Evaluate models on the RST-DT test set
        evaluator = Evaluator(model_dir='../../data/model/best')
        evaluator.pred_parser(path=args.data_dir, bcvocab=brown_clusters, draw=True)
