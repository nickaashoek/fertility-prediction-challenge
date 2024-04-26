import argparse
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

def normalize_col(col, df: pd.DataFrame):
    col_max, col_min = df[col].max(), df[col].min()
    if col_max == col_min:
        # If all the values are the same, set to 0
        df[col] = df[col].apply(lambda x: 0)
    else:
        df[col] =  df[col].apply(lambda x: (x-col_min)/(col_max-col_min))

def find_group(groups: np.array, value: float) -> float:
    """
    Find and return the group that a value belongs to
    Assumes that the groups are sorted from smallest -> largest
    We can do binary search to find the right one
    """
    idx = np.searchsorted(groups, value, side="left")
    if idx > 0 and (idx == len(groups) or math.fabs(value - groups[idx-1]) < math.fabs(value - groups[idx])):
        return groups[idx-1]
    else:
        return groups[idx]
     
def apply_groups(codebook_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Setup a dictionary that defines how the groups will be applied. Should be in place?"""
    NUM_GROUPS = 10
    for col in tqdm(cleaned_df.columns):
        if col == "nomem_encr":
            # Don't clean this column, but do preserve it
            continue
        cb_info = codebook_df[codebook_df['var_name'] == col]
        col_type = cb_info['type_var'].values[0]
        # print(col, col_type)
        """
        Don't bother with grouping categorical values, as they're already grouped
        ^ The above is actually a lie. The imputer imputes float values, so we should group those as well.
        """ 
        groups = None
        if col_type == "categorical":
            # Get the range of whole number values and group that way
            values = cleaned_df[col].unique()
            whole_values = [v for v in values if v.is_integer()]
            # TODO: We should probably normalize these values?
            groups = sorted(whole_values)
        else:
            # Normalize dates and numerics
            normalize_col(col, cleaned_df)
            # At this point, everything is normalized in the 0-1 range, so grouping is easy
            groups = np.array([0 + 1/NUM_GROUPS * i for i in range(NUM_GROUPS)])
        cleaned_df[col] = cleaned_df[col].apply(lambda x: find_group(groups, x))
    return cleaned_df

def make_groups(codebook_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    # Drop the columns we aren't using
    cleaned_df.drop(columns=["outcome_available"], inplace=True)
    # Find and apply the groups
    grouped_df = apply_groups(codebook_df, cleaned_df)
    print(grouped_df)
    grouped_df.to_csv("grouped_data.csv", index=False)

def load_data(path: str, **kwargs) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", help="Path to cleaned data", required=True)
    parser.add_argument("--codebook", help="Path to codebook", required=True)

    args = parser.parse_args()
    
    print("=== Loading ===")
    cleaned_df = load_data(args.cleaned)
    print("...loaded cleaned data")
    codebook_df = load_data(args.codebook, header=0)
    print("...loaded codebook")
    print("=== Grouping ===")
    make_groups(codebook_df, cleaned_df)