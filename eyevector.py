# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob


def boxed_to_df(fname):
    """helper function hack to meet the assumptions of future more general
     processing steps without having to solve preprocessing in general"""

    # - Get frame of box-annotated fixations here: 'MF_fix_box_ALL.df' - DONE
    df = pd.read_csv(fname,
                     sep='\t',
                     encoding='utf-8',
                     index_col=0)

    df['stimulus'] = df.filename.str[17:28]
    df = df[df.FixationIndex.notnull()]

    # - Zero-index fixations and word ids - DONE
    df.drop(['index', 'level_0'], axis=1)
    df.FixationIndex = df.FixationIndex-1
    df.token = df.token-1

    # - Build a readable unique index - DONE
    unique_index = df[['ParticipantName', 'stimulus', 'FixationIndex']
                      ].T.apply(lambda x: '_'.join([
                                str(val) for val in x.values]))
    df['unique_index'] = unique_index

    # - index for prediction
    pred_index = df[['stimulus', 'token', 'text', 'ParticipantName',
                    'FixationIndex']].T.apply(lambda x: '_'.join([
                                              str(val) for val in x.values]))
    df['pred_index'] = pred_index

    return df


def pre_snip_frame(df):
    """
    - Expects a pandas DataFrame of "box"-annotated fixations
      (e.g. from boxed_file_to_df) with:
        - Zero-indexed fixations and word ids
        - a readable unique index
    - Finds the maximal number of fixations to determine
      the maximum sequence length
    - Builds a dataframe with a column for each posible
      sequence place and both the integer- and the readable index
    """

    # - Get the maximal number of fixations to determine
    #   the maximum sequence length
    max_fixs = df.FixationIndex.max()

    # - Build a dataframe with a column for each posible
    #   sequence place
    # and both the integer- and the readable index
    snippets = pd.DataFrame(index=df.index,
                            columns=np.arange(-(max_fixs+1), max_fixs+1))
    snippets['text'] = df.text
    snippets['oov'] = df.oov
    snippets['duration'] = df.duration
    snippets['tokenID'] = df.token
    snippets['factors'] = df.factors
    snippets['fnum'] = np.NaN
    snippets['pred_index'] = df.pred_index
    snippets['unique_index'] = df.unique_index
    snippets.set_index('unique_index', append=False, inplace=True)

    # Put the fixation numbers into place
    for i, sub_df in df.groupby(['ParticipantName', 'stimulus']):
        fixnum_list = sub_df.apply(lambda x:
                                   sub_df.loc[((sub_df.token == x['token'])
                                              & (sub_df.FixationIndex
                                                 <= x['FixationIndex'])
                                               ), 'token'].count(),
                                   axis=1).values

        names = sub_df.index.map(lambda x: '_'.join([i[0], i[1],
                                 str(sub_df.loc[x, 'FixationIndex'])]))

        snippets.loc[names, 'fnum'] = fixnum_list

    return snippets


# - a function that outputs the snippet of saccade distances
#   around each fixation:
#    1. assume a person * stimulus dataframe of fixations
#       and get the token-ids
#    2. calculate relative distance for all saccades from current token
#    3. calculate jump-size and direction from previous fixation
#
# - Write a similar function that outputs similar snippets
#   with fixation durations instead of skip size.


def jump_rel(subframe, tokens, key, relative=None):
    """return array recording distances as n tokens from currently focused token
    or if relative:
    return array recording distances as n tokens from last fixation,
    i.e. jump length in tokens"""

    if relative:
        answer = (tokens-tokens.shift(1)).values

    else:
        answer = tokens.add(-subframe[subframe.FixationIndex
                            == key].token.values[0]).values

    return answer


def durs(subframe, tokens, key, relative=None):
    """return array of fixations durations
    relative should be personal means dict"""
    # part = subframe.ParticipantName.unique()[0]

    if not relative:
        result = subframe.duration.values

    else:
        result = (subframe.duration.values-relative)/relative

    return result


def snipper(subframe, repr=durs, relative=None):
    """1: assume a participant by stimulus frame of fixations,
    2: get the list of token-ids fixated from column 'token',
    3: for each fixation (row) run repr(esentation) function to get sequence,
    4: return dict of fixation-ids and its calculated sequence"""
    tokens = subframe.token
    # words = subframe.text

    sequence_dict = {key: repr(subframe, tokens, key, relative)
                     for key in subframe.FixationIndex}

    return sequence_dict


# # - function that enters a sequence into the snippet frame
# #   centered on column 0 for each individual fixation

def inserter(fix_df, snip_frame, cut=1.5, repr=durs, relative=None):
    """1: take a group id and corresponding dataframe
    2: get sequence-dict from snipper function
    3: put the sequence in the right spot of the big snip_frame"""

    for i, df in fix_df.groupby(['ParticipantName', 'stimulus']):
        if relative and repr == durs:
            subj_mean = relative[df.ParticipantName.unique()[0]]
        else:
            subj_mean = None

        if repr == jump_rel:
            seq_dict = snipper(df, repr, relative)
        else:
            seq_dict = snipper(df, repr, subj_mean)

        for key, vals in seq_dict.items():
            cols = np.arange(-key, (len(vals) - key))
            name = '_'.join([i[0], i[1], str(key)])
            if cut:
                vals = [v if np.sign(v) * v < cut
                        else np.sign(v) * cut for v in vals]

            snip_frame.loc[name, cols] = vals

    return snip_frame


# make individual first_fixations available:

def relative_dur(df):
    """To calculate scaled duration values:
    - Groups participants and keeps only a0-condition items with fnum==1
    - prints and plots
    - calculates and stores "neutral" first fixation means
    - make personal percentage deviation-based duration column"""

    personals = {}

    for part in set(df.index.map(lambda x: x.split('_')[0])):
        print(part)
        criterium = (df.index.str.contains(part + '_a0')) & (df.fnum == 1)
        personals[part] = df.loc[criterium, :].duration

    pers_means = {key: vals.mean() for key, vals in personals.items()}
    # pd.DataFrame.from_items(item for item in personals.items()
    #                        if item[0]!='P02').hist(bins=20,
    #                        figsize=(12,8),sharex=True,sharey=True)

    return pers_means


# stitch it all together to test:
def get_snippet_repr(fname='~/repos/normal_tweeting/src/MF_fix_box_ALL.df',
                     test_groups=True, jump=0, dur=0):

    fix_frame = boxed_to_df(fname)

    if test_groups:
        test_groups = ((fix_frame.ParticipantName.isin(['P04', 'P15', 'P05'])))
        # ,'P06','P04', slow ones: 'P03','P10',
        # & (fix_frame.stimulus.str.contains('a1'))
        # & (fix_frame.stimulus.str.contains('c1')))
        # & (fix_frame.stimulus.str.contains('t2'))

        fix_frame = fix_frame[test_groups]

    else:
        exclude = ['P04', 'P07', 'P13', 'P14', 'P16', 'P22', 'P24']

        fix_frame = fix_frame[~fix_frame.ParticipantName.isin(exclude)]

    snips = pre_snip_frame(fix_frame)
    if dur:
        subj_mean_dict = relative_dur(snips)
        print([i for i in subj_mean_dict.items()])
        snips = inserter(fix_frame, snips,
                         # cut=5,repr=jump_rel,relative=1
                         cut=1.5, repr=durs, relative=subj_mean_dict
                         )
    elif jump:
        snips = inserter(fix_frame, snips,
                         cut=5, repr=jump_rel, relative=1
                         # cut=1.5,repr=durs,relative=subj_mean_dict
                         )

    # NOTE: raw durs: cut 500
    # rel durs: 1.5 %
    # rel jumps: 5 tokens

    return snips


# format to Rungsted input:
def rung_former(df, fix_range=(-1, 3), fill=1, text_window=1):
    """
    Expects a dataframe of snippets [hardcoded columns!]
    Fill-na's by mean, else replace with fill val.
    outputs list of labels and vowpalwabbit formatted string.

    """
    # look_out: column names are floats :S
    fix_range = (float(i) for i in fix_range)
    # get the requested range snippets only
    fix_cols = np.arange(*fix_range).tolist()
    cols = fix_cols + ['text', 'fnum', 'oov', 'tokenID',
                       'factors', 'pred_index']
    # discard first and last fixations.
    criteria = ((df[0].notnull()) | (df[1].notnull()))
    vectors = df.loc[criteria, cols]
    # format some columns
    # vectors.oov = vectors.oov.apply(lambda x: '1' if x else '0')
    vectors.fnum = vectors.fnum.map(int)

    vectors.set_index('pred_index', append=False, inplace=True)

    # replace empty lists(!?) with a regular NaN
    vectors.loc[(vectors[0].map(type) == list), 0] = np.NaN

    # fill nans of fixation sequence cols only: fix_cols
    # fill in the rest appropriatly
    if fill == 'mean':  # callable(fill):
        # this should not be needed.... just 0 or 1
        vectors.loc[:, fix_cols] = vectors.loc[:, fix_cols
                                               ].fillna(df.duration.mean())
    else:
        vectors.loc[:, fix_cols] = vectors.loc[:, fix_cols].fillna(fill)

    labels = set()
    output = ''

    # format each line
    for j, (i, row) in enumerate(vectors.iterrows()):
        label = row['factors'] if row['oov'] else 0
        labels.add(label)
        text = str(row['text'])
        fix_num = str(row['fnum'])
        tokenID = str(row['tokenID']).strip('.0')
        pre_dists = [row[k] for k in fix_cols]
        try:
            pre_dists = [np.round(k, 2) for k in pre_dists]
        except TypeError:
            print(pre_dists, [type(i) for i in pre_dists])
        dists = ' '.join([str(k) for k in pre_dists])

        # importance = '' #str(.9 if label else .1)

        # better structure of the id:
        fix_id = i  # row['pred_index']
        # fix_id = '_'.join([i,text])

        if j % text_window == 0:
            line = "{} '{}|d {} |n fnum={} |t token={} tokenID={} \n\n".format(
                label, fix_id, dists, fix_num, text, tokenID)
        else:
            line = "{} '{}|d {} |n fnum={} |t token={} tokenID={} \n".format(
                label, fix_id, dists, fix_num, text, tokenID)

        output += line
    return output, labels


def data_split(df, sizes=(.8, .1), text_window=1, fill=1, fix_range=(-1, 3)):
    """
    splits a dataframe into train, dev and test set.
    """

    print('test part will be:', 1-sum(sizes))

    train_size = round(len(df)*sizes[0])
    dev_size = round(len(df)*sizes[1])

    np.random.seed(1)
    shuffled_seq = np.random.permutation(len(df))

    train_set = shuffled_seq[:train_size]
    dev_set = shuffled_seq[train_size:train_size+dev_size]
    test_set = shuffled_seq[train_size+dev_size:]

    to_rungsted_learn, labels = rung_former(df.iloc[train_set],
                                            text_window=text_window,
                                            fill=fill,
                                            fix_range=fix_range)
    to_rungsted_dev, dev_labels = rung_former(df.iloc[dev_set],
                                              text_window=text_window,
                                              fill=fill,
                                              fix_range=fix_range)
    to_rungsted_test, test_labels = rung_former(df.iloc[test_set],
                                                text_window=text_window,
                                                fill=fill,
                                                fix_range=fix_range)

    to_rungsted_learn = to_rungsted_learn.replace(':', '[..]')
    to_rungsted_dev = to_rungsted_dev.replace(':', '[..]')
    to_rungsted_test = to_rungsted_test.replace(':', '[..]')

    print('size of train, dev and test:',
          len(train_set), len(dev_set), len(test_set))
    print('labelsets:\n{}\n{}\n{}'.format(labels, dev_labels, test_labels))

    return to_rungsted_learn, to_rungsted_dev, to_rungsted_test, labels, dev_labels, test_labels


def parse_preds(predictions):
    """
    Read the Rungsted prediction output and build dataframe with
    one row per word and multiindexed columns per fixation snip per
    participant and one column with the true label.
    Return raw parsed predictions
    """
    preds = pd.read_csv(predictions, encoding='utf-8',
                        sep='\t',  # escapechar="b'",#verbose=True,
                        converters={label: lambda x: x.lstrip("b'")
                                    .rstrip("'").strip('"') for label
                                    in ['fixID', 'label', 'prediction']},
                        skip_blank_lines=True, header=None,
                        names=['fixID', 'label', 'prediction'])

    # id: 'stimulus','token','text_with_underscores',
    # 'ParticipantName','FixationIndex'
    preds.fixID = preds.fixID.str.replace('__', '_')
    preds['stimulus'] = preds.fixID.apply(lambda x: '_'.join(x.split('_')[:2]))
    preds['token'] = preds.fixID.apply(lambda x: float(x.split('_')[2:3][0]))
    preds['fixation'] = preds.fixID.apply(lambda x:
                                          float(x.split('_')[-2:-1][0]))
    preds['fnum'] = preds.fixID.apply(lambda x: float(x.split('_')[-1:][0]))
    preds['participant'] = preds.fixID.apply(lambda x:
                                             '_'.join(x.split('_')[-3:-2]))
    preds['text'] = preds.fixID.apply(lambda x: '_'.join(x.split('_')[3:-3]))

    # calculate results.
    correct = (preds.prediction == preds.label)
    preds['result'] = correct
    preds['mfs'] = '0' == preds.label
    preds['bina'] = (preds.prediction
                     == preds.label) | ((preds.prediction != '0')
                                        & (preds.label != '0'))

    return preds


def importance_weight(label='a0c0t0p0', fnum=1, i='', maj_cost=.1, a2cost=1):
    """importance weighting by 1/fnum.
    Downweight 0's (majority) by maj_cost.
    Downweight a2-labels by a2cost."""

    if float(fnum) == 0:
        weight = 0.

    else:
        weight = 1/(float(fnum))

    if label == 0:
        weight = maj_cost*weight

    elif str(label).startswith('a2'):
        weight = a2cost*weight

    return str(weight)


# format 2 dfs to Rungsted input:
def rung_former2(df_list, fix_range=(-1, 3), fill_vals=[0, 1], text_window=1):
    """
    Expects a list of dataframes of snippets [hardcoded columns!]
    Fill-na's by fill vals, respectively.
    outputs list of labels and vowpalwabbit formatted string.

    """
    # look_out: column names are floats :S
    fix_cols = [float(i) for i in np.arange(*fix_range).tolist()]
    cols = fix_cols + ['text', 'fnum', 'oov',
                       'tokenID', 'factors', 'pred_index']

    vect_dfs = []

    for fill, df in zip(fill_vals, df_list):
        # discard first and last fixations.
        criteria = ((df[0].notnull()) | (df[1].notnull()))
        vectors = df.loc[criteria, cols]

        vectors.fnum = vectors.fnum.map(int)

        vectors.set_index('pred_index', append=False, inplace=True)

        # make fill avg if dur.
        if fill == 0:
            fill = vectors.loc[:, 0.0].mode().values[0]
            print(fill)

        # replace empty lists(!?) with fill val
        vectors.loc[(vectors[0].map(type) == list), 0] = fill

        # fill nans of fixation sequence cols only: fix_cols
        # fill in the rest appropriatly

        vectors.loc[:, fix_cols] = vectors.loc[:, fix_cols].fillna(fill)
        vect_dfs.append(vectors)

    labels = set()
    output = ''

    # format each line
    for j, (i, row) in enumerate(vect_dfs[0].join(vect_dfs[1],
                                 rsuffix='fix', lsuffix='dur').iterrows()):
        # label = '1' if row['oovdur'] else 0
        label = row['factorsdur'] if row['oovdur'] else 0
        if label:
            factors = ['c2', 'c1', 't2', 'p2']
            lab_facs = [label[:2], label[2:4], label[4:6], label[6:]]
            highs = [h for h in lab_facs if h in factors]
            label = ''.join(highs)
        labels.add(label)
        text = str(row['textdur'])
        fix_num = str(row['fnumdur'])
        tokenID = str(row['tokenIDdur']).strip('.0')

        pre_fix_cols = [str(c)+'dur' for c in fix_cols]
        pre_dists = [row[k] for k in pre_fix_cols]
        pre_dists = [np.round(k, 2) for k in pre_dists]
        dur_dists = ' '.join([str(k) for k in pre_dists])

        pre_fix_cols = [str(c)+'fix' for c in fix_cols]
        pre_dists = [row[k] for k in pre_fix_cols]
        pre_dists = [np.round(k, 0) for k in pre_dists]
        fix_dists = ' '.join([str(k) for k in pre_dists])

        fix_id = '_'.join([i, fix_num])

        importance = importance_weight(label, fix_num, i)

        if j % text_window == 0:
            line_end = '\n\n'
        else:
            line_end = '\n'

        line = "{} {} '{}|d {} |f {} |n fnum={} |t token={} |i tokenID={} {}"\
            .format(label, importance, fix_id, dur_dists,
                    fix_dists, fix_num, text, tokenID, line_end)

        output += line
    return output, labels


def data_split2(df_list,
                sizes=(.8, .1),
                text_window=1,
                fill_vals=[0., 1.],
                fix_range=(-1, 3)):
    """
    splits a list of dataframes into train, dev and test set.
    """

    df_size = len(df_list[0])

    # set the sizes (test set will get the rest.)
    print('test part will be:', 1-sum(sizes))
    train_size = round(df_size*sizes[0])
    dev_size = round(df_size*sizes[1])

    # FIX ME! should not shuffle randomly, but by text, participant,
    # and chunck-size too...
    np.random.seed(1)
    shuffled_seq = np.random.permutation(df_size)

    train_set = shuffled_seq[:train_size]
    dev_set = shuffled_seq[train_size:train_size+dev_size]
    test_set = shuffled_seq[train_size+dev_size:]

    # pick-out the relevant dataset for train, dev and test
    to_rungsted_learn, labels = rung_former2([df_list[0].iloc[train_set],
                                             df_list[1].iloc[train_set]],
                                             text_window=text_window,
                                             fill_vals=fill_vals,
                                             fix_range=fix_range)
    to_rungsted_dev, dev_labels = rung_former2([df_list[0].iloc[dev_set],
                                               df_list[1].iloc[dev_set]],
                                               text_window=text_window,
                                               fill_vals=fill_vals,
                                               fix_range=fix_range)
    to_rungsted_test, test_labels = rung_former2([df_list[0].iloc[test_set],
                                                 df_list[1].iloc[test_set]],
                                                 text_window=text_window,
                                                 fill_vals=fill_vals,
                                                 fix_range=fix_range)

    # handle problem with colon as special character
    to_rungsted_learn = to_rungsted_learn.replace(':', '[..]')
    to_rungsted_dev = to_rungsted_dev.replace(':', '[..]')
    to_rungsted_test = to_rungsted_test.replace(':', '[..]')

    # inspect that sizes are reasonable
    print('size of train, dev and test:',
          len(train_set), len(dev_set), len(test_set))
    print('labelsets:\n{}\n{}\n{}'.format(labels, dev_labels, test_labels))

    # data and label sets
    return (to_rungsted_learn, to_rungsted_dev, to_rungsted_test,
            labels, dev_labels, test_labels)


def displayer(fixID,
              stim_prefix='/Users/sigrid/repos/normal_tweeting/src/stimuli/'):
    # a1c1t1p2_7_9.0_Start_P15_11.0_3
    # is at /Users/sigrid/repos/normal_tweeting/src/stimuli/
    # a1c1t1p2/a1c1t1p2_7*.csv
    folder, fprefix = fixID.split('_')[:2]
    filename = glob.glob(stim_prefix + folder + '/'
                         + folder + '_' + fprefix + '_*.tsv')[0]
    boxes = pd.read_csv(filename, sep='\t', encoding='utf-8', header=0)
    boxes['w_id'] = boxes['id'].apply(lambda x: int(x.split('-')[0])-1)
    return boxes.groupby('w_id').agg({'text': ''.join}).T
