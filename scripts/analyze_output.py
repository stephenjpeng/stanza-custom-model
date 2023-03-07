import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='output/output.tsv', help='Location of eval output')
    parser.add_argument('--wrong_sents_file', type=str, default=None, help='Location to save wrong sentences')
    args = parser.parse_args()

    args = vars(args)
    return args

def load_output(fname):
    df = pd.read_csv(fname, sep='\t', header=None, index_col=False,
            skip_blank_lines=False)
    df.columns = ['word', 'gold', 'pred']
    df = df.drop(df.shape[0] - 1, axis=0)

    df['sentence'] = np.cumsum(pd.isna(df.word))
    df = df.dropna(subset=['word'])

    df = df.groupby('sentence').agg(' '.join)
    df['correct'] = df.gold == df.pred

    return df

if __name__=="__main__":
    args = parse_args()
    output = load_output(args['output_file'])
    if args['wrong_sents_file']:
        output[~output.correct].to_csv(args['wrong_sents_file'], index=False)
    else:
        print(output[~output.correct])

