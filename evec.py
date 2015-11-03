# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from string import printable
import urllib.parse
import re


# dummy_fs = ['../data_export/Logic_reading_Data_Export (2).tsv',
# '../data_export/Logic_reading_Data_Export (3).tsv',
# '../data_export/Logic_reading_Data_Export (4).tsv',
# '../data_export/Logic_reading_Data_Export (5).tsv',
# '../data_export/Logic_reading_Data_Export (6).tsv']


def prettify(fname, export_cols=None):
    """Remove non-printing chars from raw tobii-generated .tsv"""
    lines = open(fname, 'r', encoding='utf-8').readlines()
    temp_fn = 'temp_clean_data.tsv'
    clean_file = open(temp_fn, 'w', encoding='utf-8')
    labels = ''.join([c for c in lines[0] if c in printable])
    clean_file.write(labels)
    clean_file.write(''.join(lines[1:]))
    clean_file.close()
    # make dataframe
    f = pd.read_csv(temp_fn, sep=u'\t', usecols=export_cols,
                    encoding='utf-8', low_memory=False)
    return f


# Formatting functions (destinct per experiment):
def factor_parser(media_name):
    """media names are turned into bools, tID is kept a string,
    line is returned as an int"""
    facts, line, tID = media_name.split('_')
    tID = tID.rstrip('.png')
    a, c, t, p = [int(c) for c in facts if c.isdigit()]
    line = int(line)
    return (tID, line, a, c, t, p, facts)


def logic_parser(media_name):
    """media names are turned into url and query"""
    if type(media_name) != str:
        return ('', '')
    else:
        media = urllib.parse.urlsplit(media_name)
        path = media.path
        query = media.query
        return (path, query)


def between_clean(df):
    df.StudioEvent = df.StudioEvent.ffill()
    df = df[df.StudioEvent != 'URLEnd']
    return df


def __screener__(row):
    screen = ('cross' if row.url == '/sigrid/cross.php'
              else 'compa' if row.url == '/sigrid/quality.php'
              else 'solve' if ((row.url == '/sigrid/text.php')
                               & (row.phpquery > ''))
              else 'compr' if ((row.url == '/sigrid/text.php')
                               & ~(row.phpquery > ''))
              else np.NaN)
    return screen


def early_data(df, print_times=True, save_bl_frame=True):
    """
    remove irrelevant early data per participant.
    Use groupby instead of for-loops.
    fills additional rows
    save_bl_frame saves the baseline reading data under that variable name.
    """

    # distinguish the tasks in dedicated column
    df['Screen'] = df[['url', 'phpquery']].apply(lambda x: __screener__(x),
                                                 axis=1)

    # recognize start and end of trials
    test_start = '/sigrid/text.php'
    test_end = '/sigrid/quality.php'

    grouped = df.groupby(['ParticipantName', 'RecordingName'])

    cleaned_idx = []
    bl_idx = []
    for (subj, rec), sub_df in grouped:
        time_start = sub_df[((sub_df.url == test_start)
                             & (sub_df.StudioEvent == 'URLStart'))
                            ].RecordingTimestamp
        time_end = sub_df[((sub_df.url == test_end)
                           & (sub_df.StudioEvent == 'URLEnd'))
                          ].RecordingTimestamp

        if rec == 'First':
            # skip 3 demos (text.php is visited twice per trial)
            clean_df = sub_df[sub_df.RecordingTimestamp >= time_start.iloc[6]]
            bl = pd.DataFrame()  # make it empty and check later if it's empty
        else:
            # skip 1 baseline (as above)
            clean_df = sub_df[sub_df.RecordingTimestamp >= time_start.iloc[2]]
            bl = sub_df[((sub_df.RecordingTimestamp > time_start.iloc[0])
                        & (sub_df.RecordingTimestamp <= time_start.iloc[2]))]
            bl_idx += bl.index.tolist()

        clean_df = clean_df[clean_df.RecordingTimestamp <= time_end.iloc[-1]]

        if print_times:
            print(subj, rec)
            start_time = clean_df.RecordingTimestamp.iloc[0]/1000/60.
            end_time = clean_df.RecordingTimestamp.iloc[-1]/1000/60.
            print('Recording time spent:', end_time - start_time)
            if len(time_start.tolist()) < 42:
                print("this recording has suspiciously few items:",
                      len(time_start.tolist()))
            if not bl.empty:
                print('Baseline time spent:',
                      (bl.RecordingTimestamp.iloc[-1]/1000/60.)
                      - (bl.RecordingTimestamp.iloc[0]/1000/60.))

        # actually, just keep the index to cut out relevant bits...
        cleaned_idx += clean_df.index.tolist()

        # NOW, FILL IN STUFF:
        clean_df.Screen.ffill(inplace=True)
        clean_df.Manipulation.bfill(inplace=True)
        clean_df.Manipulation.ffill(inplace=True)
        clean_df.QuestionName.bfill(inplace=True)
        clean_df.QuestionName.ffill(inplace=True)
        df[(df.ParticipantName == subj) & (df.RecordingName == rec)] = clean_df

        if save_bl_frame:
            # crop rows between screens
            bl_df = between_clean(df.loc[bl_idx, :])
            # save it
            bl_df.to_csv(subj+'_baseline.tsv',
                         sep='\t', encoding='utf-8')

    # crop rows between screens
    df = between_clean(df.loc[cleaned_idx, :])

    return df


# helper functions for smt_data
def __qname__(letters):
    if type(letters) == str and "tasks" in letters:
        parsed = urllib.parse.urlsplit(letters)
        parts = urllib.parse.parse_qsl(parsed.query)
        # this gets a list of tuples:
        # example: [(u'file', u'tasks/DAhuman462Infe.png'), (u'correctAnswer',
        # u'2'), (u'answer', u'2')]
        if len(parts) > 1:
            idx = [(m.start(0)) for m in
                   re.finditer('[0-9][0-9][0-9]', parts[0][1])]
            return parts[0][1][idx[0]:]  # fix dette og fjern .png


def __manip__(letters):
    if type(letters) == str and "http" in letters:
        parsed = urllib.parse.urlsplit(letters)
        # this gets a list of tuples
        parts = urllib.parse.parse_qsl(parsed.query)
        # example: [(u'file', u'tasks/DAhuman462Infe.png'), (u'correctAnswer',
        # u'2'), (u'answer', u'2')]
        if len(parts) > 1:
            if 'DAhuman' in parts[0][1]:
                manip = 'DAhuman'
            elif 'DAsmt' in parts[0][1]:
                manip = 'DAsmt'
            elif 'DAsimsmt'in parts[0][1]:
                manip = 'DAsimsmt'
            elif 'ENorig' in parts[0][1]:
                manip = 'ENorig'
            elif 'ENhuman' in parts[0][1]:
                manip = 'ENhuman'
            else:
                manip = ''
            return manip


def __correct__(url):
    if type(url) == str and "correctAnswer=" in url:
        parsed = urllib.parse.urlsplit(url)
        parts = urllib.parse.parse_qsl(parsed.query)
        # this gets a list of tuples:
        # example [('file',
        # 'tasks/DAhuman381Eval.png'),('correctAnswer', '1'),('answer', '3')]
        return parts[1][1] == parts[2][1]


def __compr__(letters):
    if type(letters) == str:
        if "comprehensible=" in letters:
            comp_idx = letters.index("comprehensible=")+15
            return int(letters[comp_idx])


def __compare__(url):
    if type(url) == str and u"cross" in url and u"tasks" in url:
        parsed = urllib.parse.urlsplit(url)
        # this gets a list of tuples
        parts = urllib.parse.parse_qsl(parsed.query)
        # example [('topFile', 'tasks/ENorig361Eval.png'), ('bottomFile',
        # 'tasks/ENhuman361Eval.png'), ('rating', '4')]
        # hack to avoid missing ratings:
        if len(parts) == 3:
            lower = parts[1][1][6:]
            rating = int(parts[2][1])
            return (lower, rating)


def add_cols_smt(df):
    df['QuestionName'] = df.StudioEventData.map(__qname__)
    df['Manipulation'] = df.StudioEventData[((df.StudioEvent
                                             == "URLStart"))].map(__manip__)
    df['CorrectAns'] = df.StudioEventData[((df.StudioEvent
                                           == "URLStart"))].map(__correct__)
    df['Comprehensibility'] = df.StudioEventData[((df.StudioEvent ==
                                                  "URLStart"))].map(__compr__)
    df['CompareBottom'] = df.StudioEventData.map(__compare__)
    df['CompareRating'] = df.CompareBottom.apply(lambda x:
                                                 x[1] if (type(x) == tuple)
                                                 else x)
    df['CompareBottom'] = df.CompareBottom.apply(lambda x:
                                                 x[0] if (type(x) == tuple)
                                                 else x)

    return df


def tobii_importer(fnames, format_func={'StudioEventData':
                                        (logic_parser,
                                         ['url', 'phpquery'])},
                   discard=['MKK', 'PLP', 'GTC'],
                   cleaner_func=early_data,
                   add_cols_func=add_cols_smt,
                   save_import_as=None):
    """Expects a list of filenames containing tobii-csv's to be combined.
    Format_func should be a dictionary with column names as keys
    and touples of functions and outputs  as values.
    Cleaner_func is for discarding experiment specific
    irrelevant bits of the frame"""
    export_cols = ['ParticipantName', 'RecordingName', 'MediaName',
                   'RecordingTimestamp', 'EyeTrackerTimestamp',
                   'StudioEventIndex', 'StudioEvent', 'StudioEventData',
                   'FixationIndex', 'SaccadeIndex',
                   'GazeEventType', 'GazeEventDuration',
                   'FixationPointX (MCSpx)', 'FixationPointY (MCSpx)',
                   'GazePointIndex',
                   'GazePointX (ADCSpx)', 'GazePointY (ADCSpx)']

    new_cols = [s.replace(' ', '_').replace('(', '').replace(')', '')
                for s in export_cols]

    all_subjs = []
    master_df = pd.DataFrame()

    for f in fnames:
        pretty_df = prettify(f, export_cols)
        pretty_df.columns = new_cols

        # in case exports include the same participants,
        # only include new ones here
        new_subjs = [p for p in pretty_df.ParticipantName.unique()
                     if (p not in all_subjs) and (p not in discard)]
        if len(new_subjs):
            all_subjs += new_subjs

            pretty_df = pretty_df[pretty_df.ParticipantName.isin(new_subjs)]

            for col, (func, outputs) in format_func.items():
                result_col = pretty_df[col].map(func)
                for i, output in enumerate(outputs):
                    pretty_df[output] = result_col.apply(lambda x:
                                                         x[i])

            # this may not be too relevant after all..?
            if len(master_df) == 0:
                master_df = pretty_df
            else:
                master_df = master_df.append(pretty_df,
                                             ignore_index=True)

    # Make recording-field title-cased
    master_df.RecordingName = master_df.RecordingName.str.title()
    master_df = master_df.reset_index(drop=True)
    # Remove 3K+ rows of no-info
    master_df = master_df[master_df.ParticipantName.notnull()]

    if add_cols_func:
        master_df = add_cols_func(master_df)

    if cleaner_func:
        master_df = cleaner_func(master_df)

    if save_import_as:
        master_df.to_csv(save_import_as,
                         sep='\t', encoding='utf-8')

    return master_df
