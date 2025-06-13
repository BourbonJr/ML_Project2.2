import pandas as pd
import numpy as np
import yaml
import argparse

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def select_features(df):  
    v1_features = [
        'I1', 'I2', 'I3',
        'gx', 'gy', 'gz',
        'ax', 'ay', 'az',
        'V1real', 'V2real', 'V3real',
        'N1', 'N2', 'N3'
    ]
    
    selected_features = v1_features 
    return df[selected_features]

def generate_binary_target(df):
    return (df['Type'] == 3).astype(int)

def process_and_save(df, config, output_path):
    df.columns = df.columns.str.strip()
    
    X = select_features(df)
    y = generate_binary_target(df)

    result_df = X.copy()
    result_df['Type'] = y.values
    result_df.to_csv(output_path, index=False)

def main(config_path):
    config = load_config(config_path)

    train_df = pd.read_excel(config['data_load']['train_dataset']).dropna()

    process_and_save(
        train_df,
        config,
        config['data_split']['trainset_path']
    )

    test_df = pd.read_excel(config['data_load']['test_dataset']).dropna()

    process_and_save(
        test_df,
        config,
        config['data_split']['testset_path']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
