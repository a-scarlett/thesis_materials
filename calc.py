import pandas as pd
import json
import pickle
import argparse
import os


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        raise


def process_data(data):
    df = pd.json_normalize(data, 'methodFeatures', ['file'])
    names = df['name']
    df['index'] = df.index + 1
    df['sum_feat'] = (df['entropy'] + df['halstead']) / df['lines']
    df['mult_feat'] = (df['entropy'] * df['halstead']) / df['lines']

    df_agg = df.groupby('index').agg({
        'lines': ['min', 'max', 'mean', 'count'],
        'maxNesting': ['min', 'max', 'mean'],
        'maxAstDistance': ['min', 'max', 'mean'],
        'halstead': ['min', 'max', 'mean'],
        'entropy': ['min', 'max', 'mean'],
        'sum_feat': ['min', 'max', 'mean'],
        'mult_feat': ['min', 'max', 'mean']
    })

    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]

    columns_to_transform = [
        'lines_min', 'lines_max', 'lines_mean', 'lines_count',
        'maxNesting_min', 'maxNesting_max', 'maxNesting_mean',
        'maxAstDistance_min', 'maxAstDistance_max', 'maxAstDistance_mean',
        'halstead_min', 'halstead_max', 'halstead_mean', 'entropy_min',
        'entropy_max', 'entropy_mean'
    ]

    for col in columns_to_transform:
        df_agg[f'transf_{col}'] = df_agg[col] / (df_agg[col] + 1)

    df_to_emb = df_agg[['sum_feat_mean',
                        'transf_maxAstDistance_max',
                        'mult_feat_max',
                        'sum_feat_min',
                        'transf_halstead_mean',
                        'transf_entropy_mean',
                        'transf_maxAstDistance_min',
                        'sum_feat_max',
                        'transf_lines_mean']]

    return names, df_agg, df_to_emb


def main():
    parser = argparse.ArgumentParser(description="Please provide path to json with input data")
    parser.add_argument('path', type=str, help='Path to the JSON file')
    args = parser.parse_args()

    input_path = args.path
    output_path = os.path.splitext(input_path)[0] + '.csv'

    data = load_json_file(input_path)

    names, df, df_to_emb = process_data(data)
    with open('tsne.pkl', 'rb') as file:
        tsne = pickle.load(file)
    embedded = tsne.transform(df_to_emb)
    df['tsne-x'] = embedded[:, 0]
    df['tsne-y'] = embedded[:, 1]
    df['tsne-z'] = embedded[:, 2]

    features_to_train = ['transf_maxAstDistance_max',
                         'transf_maxNesting_min',
                         'tsne-x',
                         'sum_feat_min',
                         'tsne-y',
                         'transf_maxAstDistance_mean',
                         'transf_maxNesting_max',
                         'transf_lines_mean',
                         'sum_feat_max',
                         'sum_feat_mean',
                         'transf_maxNesting_mean']

    df_prep = df[features_to_train]

    with open('random_forest.pkl', 'rb') as file:
        rf = pickle.load(file)
    predictions = rf.predict(df_prep)

    res = pd.DataFrame({
        'name': names,
        'res': predictions
    })
    res.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
