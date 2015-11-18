# -*- coding: utf-8 -*-

# import numpy as np
import pandas as pd
import os

# Expects dataframe as output by tobii-importer and produces
# AOI-boxed fixation-based dataframe

# Note that annotation format of aoi's is task-dependent


# Collect fixation durations and keep only one row per fixation
def fix_durs(df):
    duration = df.RecordingTimestamp.max() - df.RecordingTimestamp.min()
    return duration


def smt_errs(annotations='../predict_errors/error_annotation.csv'):
    """hacky helper function to handle the particular format
    of smt-error annotations"""
    smt_err = pd.read_csv(annotations,
                          sep=';',
                          index_col=0,
                          usecols=[0, 1, 2, 4, 5, 6, 7],
                          header=0,
                          names=['aoi_index', 'q_id', 
                                 'w_id', 'w_missing',
                                 'w_order', 'w_incorr', 'w_unk'])
    smt_err = smt_err.fillna(0)
    smt_err['w_total'] = (smt_err.w_missing + smt_err.w_order +
                          smt_err.w_incorr + smt_err.w_unk)

    smt_err['w_binary'] = smt_err.w_total.apply(lambda x: 1 if x != 0
                                                else 0)

    smt_err['w_cat'] = smt_err.apply(lambda x: 'Incorrect' if x.w_incorr
                                     else 'Missing' if x.w_missing
                                     else 'Order' if x.w_order
                                     else 'Unknown' if x.w_unk
                                     else 'Ok', axis=1)
    return smt_err


def smt_boxes(box_path='../logic_reading/experiment_copy/stimuli/'):
    """hacky helper function to pick out relevant files only"""
    smt_files = [box_path + i for i in os.listdir(box_path)
                 if ('.tsv' in i)
                 # and ('DA' in i)
                 # and (('Infe' in i) or ('Conc' in i))
                 and ('.question' not in i)]
    return smt_files


def boxing(tsv_path):
    # make frame of aoi-.tsv
    q_id = tsv_path.split('/')[-1][:-4]
    frame = pd.read_csv(tsv_path, encoding='utf-8', header=0, sep='\t')
    frame['q_id'] = q_id

    # group by w_id
    frame['w_id'] = frame['id'].apply(lambda x: int(x.split('-')[0]))
    grouper = frame.groupby('w_id')

    # build new frame with one row per grouped item using aggregate
    boxes = grouper.agg({'q_id': lambda x: x.iloc[0],
                         'w_id': lambda x: x.iloc[0],
                         'text': lambda x: x.sum(),
                         # 'w_len': lambda x: x.count(),
                         'left': lambda x: x.min(),
                         'right': lambda x: x.max(),
                         'top': lambda x: x.iloc[0],
                         'bottom': lambda x: x.iloc[0]})

    return boxes


def boxset(box_files=smt_boxes(),
           annotations=smt_errs()):
    """Make a dataframe out of individual csv-files of letter-boxes
    and some annotation"""
    tsvs = [boxing(f) for f in box_files]
    boxed = pd.concat(tsvs)
    boxed = boxed.reset_index(drop=True)

    # TODO do something with annotations here...
    # annot_boxes = pd.concat([boxed, annotations], 1)
    annot_boxes = pd.merge(boxed, annotations,
                           how='outer', on=['w_id', 'q_id'])
    return annot_boxes


# Annotate one fixation with according box
def boxer(fix_row, annot='w_cat', boxframe=boxset()):
    """Return the word id, the text and the annotation
    that a fixation landed on, if any."""
    # retrieve relevant info about fixation
    fname = fix_row.filename

    x = fix_row.FixationPointX_MCSpx
    y = fix_row.FixationPointY_MCSpx
    # get the token-id covering the fixation
    f_box = boxframe[boxframe.q_id == fname]

    # lookout: y is centered at top corner. - make floats to compare to NaNs
    criterium = ((f_box.left*1. < x) & (x < f_box.right*1.)
                 & ((f_box.top*1. < y) & (y < f_box.bottom*1.)))
    box = f_box[criterium]
    # check if there are any hits
    if box.text.count() > 0:
        box_token = box.w_id.values[0]
        word = box.text.values[0]
        label = box[annot].values[0]
        result = (box_token, word, label)
    else:
        result = ('NA', 'NA', 'NA')

    return result


def drop_smt_stuff(df, boxes=boxset()):
    # only reading
    df = df[df.Screen == 'compr']

    # include only rows classified as fixations
    df = df[df.FixationIndex.notnull()]
    cols = ['MediaName', 'ParticipantName', 'FixationIndex']
    durations = df.groupby(cols).apply(fix_durs)
    df = df.drop_duplicates(cols)
    durations.name = 'duration'
    new_df = df.join(durations, on=cols, rsuffix='_new')
    # new_df.drop('duration_new', axis=1)
    new_df['filename'] = (new_df.Manipulation
                          + new_df.QuestionName.str.strip('.png'))

    new_df.loc[:, "dummy"] = None
    collected = new_df.apply(lambda x: boxer(x, boxframe=boxes), axis=1)
    new_df.loc[collected.index, ["dummy"]] = collected

    new_df.loc[:, "token"] = new_df.dummy.apply(lambda x: x[0])
    new_df.loc[:, "text"] = new_df.dummy.apply(lambda x: x[1])
    new_df.loc[:, "annotated"] = new_df.dummy.apply(lambda x: x[2])

    return new_df
