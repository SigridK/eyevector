# -*- coding: utf-8 -*-

# import numpy as np
# from sklearn.cross_validation import LeaveOneLabelOut
# import pandas as pd
import random

def cv_split_on_label(df, label_col, shuffle_seed=1):
    """generator for yielding len(label)-1 pairs of
    training and test-sets as dfs
    where one label is held out for test.
    Shuffling stimuli if shuffle_seed is not False"""
    
    labels = df.label_col.unique()
    
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(labels)

    for label in labels:
        yield df[df[label_col] == label], df[df[label_col] != label]
