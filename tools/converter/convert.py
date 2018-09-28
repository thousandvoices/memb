import pandas as pd
import numpy as np
import argparse
import csv

from memb import Builder, available_compression_strategies


def convert_fasttext(filename):
    with open(filename) as f:
        nrows, dim = map(int, f.readline().split())

    df = pd.read_csv(
        filename,
        delimiter=' ',
        header=None,
        index_col=0,
        skiprows=1,
        quoting=csv.QUOTE_NONE)

    return df, dim


def convert_glove(filename):
    df = pd.read_csv(
        filename,
        delimiter=' ',
        header=None,
        index_col=0,
        quoting=csv.QUOTE_NONE)

    dim = len(df.columns)

    return df, dim


FILE_CONVERTERS = {
    'fasttext': convert_fasttext,
    'glove': convert_glove
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert word vectors to binary format.')
    parser.add_argument('--from', dest='source_filename', required=True)
    parser.add_argument('--to', dest='dest_filename', required=True)
    parser.add_argument('--compression', dest='compression', required=True, choices=available_compression_strategies())
    parser.add_argument('--converter', dest='converter', required=True, choices=FILE_CONVERTERS.keys())
    args = parser.parse_args()

    converter = FILE_CONVERTERS[args.converter]
    embedding_df, dim = converter(args.source_filename)
    builder = Builder(dim, args.compression)

    for idx, row in embedding_df.iterrows():
        word = str(row.name)
        try:
            builder.add_word(word, np.float32(row.values[:dim]))
        except Exception as e:
            print('Exception ({}) while trying to add word'.format(e))

    builder.save(args.dest_filename)
