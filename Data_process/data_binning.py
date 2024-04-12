import pandas as pd
import pickle
import random
import warnings

# Disable warnings to keep the output clean
warnings.filterwarnings("ignore")


def data_read(file_list):
    """Reads lines of data from a list of files.

    Args:
      file_list: A list of file paths to be read.

    Returns:
      A list of strings, each representing a stripped line of data read from the files.
    """
    data = []
    for file_path in file_list:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(line.strip())
    return data


def randomly_sample_lines(file_list, sample_fraction=0.1):
    """Randomly samples lines from files listed in `file_list` with a given sampling fraction.

    Args:
      file_list: A list of file paths to sample from.
      sample_fraction: Fraction of lines to sample from each file. Defaults to 0.1.

    Returns:
      A list of sampled lines from the files.
    """
    sampled_lines = []
    for file_path in file_list:
        print(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                if random.random() < sample_fraction:
                    sampled_lines.append(line)
    return sampled_lines


def data_to_df(data, columns):
    """Converts a list of tab-separated strings to a pandas DataFrame with specified columns.

    Args:
      data: A list of tab-separated strings.
      columns: A list of column names for the DataFrame.

    Returns:
      A pandas DataFrame created from the input data.
    """
    print(len(data))
    split_data = [item.split('\t') for item in data]
    df = pd.DataFrame(split_data, columns=columns)
    return df


def main():
    """Main function to execute the script logic."""
    # Example file paths
    file_list = ['path/to/your/datafile1.txt', 'path/to/your/datafile2.txt']
    sampled_lines = randomly_sample_lines(file_list, sample_fraction=0.02)

    # Example column names
    columns = ["column1", "column2", "column3"]

    df = data_to_df(sampled_lines, columns)
    
    numeric_feature = "column2"  # Assuming 'column2' is the feature to be selected
    dz_columns_num = [name.strip() for name in numeric_feature.split(',')]
    df_dz_numeric = df[dz_columns_num]

    # Example number bins
    num_bins_list = [8, 16, 32, 64]

    for num_bins in num_bins_list:
        bin_edges_dict_dz_numeric = {}
        bin_edges_dict_person_numeric = {} # the same as bin_edges_dict_dz_numeric
        for col in df_dz_numeric.columns:
            if col == 'cfrnid':  # Assuming 'cfrnid' is a column that should be skipped
                continue
            flat_series = df_dz_numeric[col].str.split(',', expand=True).stack().astype(float).tolist()
            try:
                bins_frequency = pd.qcut(flat_series, q=num_bins, duplicates='drop')
                bin_edges_frequency = bins_frequency.categories
                bin_edges_dict_dz_numeric[col] = bin_edges_frequency.right
            except ValueError as e:
                print(f'Error while processing column {col}: {e}')
                bin_edges_dict_dz_numeric[col] = []

        print(f'Bin edges for {num_bins} bins:', bin_edges_dict_dz_numeric)

        filename = f'{num_bins}bin.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(bin_edges_dict_dz_numeric, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return bin_edges_dict_dz_numeric, bin_edges_dict_person_numeric


if __name__ == "__main__":
    bin_edges_dict_dz_numeric, bin_edges_dict_person_numeric = main()
