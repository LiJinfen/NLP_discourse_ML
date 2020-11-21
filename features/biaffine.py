#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Jinfen Li


from allennlp.predictors.biaffine_dependency_parser import Predictor
import numpy as np
predictor = Predictor.from_path("biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

def biaffine_features(sentence):
  with predictor.capture_model_internals() as internals:
    predictor.predict_batch_json([{"sentence":"ia m"},{"sentence":"ia m"}])
    # predictor.predict_json({"sentence":sentence})
  print(np.average(np.array(internals[26]['output']),axis=1).shape)
  # print([np.average(np.array(internals[26]['output']),axis=1)[0],np.average(np.array(internals[31]['output']),axis=1)[0],np.average(np.array(internals[45]['output']),axis=1)[0]])
  return np.concatenate([np.array(internals[26]['output']),np.array(internals[31]['output']),np.array(internals[45]['output'])],axis=2)

print(biaffine_features("I am hungry ").shape)