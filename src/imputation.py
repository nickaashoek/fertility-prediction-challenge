from datetime import datetime
from functools import reduce
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from sklearn.impute import KNNImputer

import argparse

######################## GLOBALS ########################

DATE_FORMATS = (
    '%d.%m.%Y',
    '%d-%m-%Y',
    '%d/%m/%Y',
)
NULL_STRINGS = {"nan", "NAN", "NaN", "null", "Null", "NULL", ""}


######################## COMMON ########################

def _convert(col: np.array, convert_fn) -> np.array:
    '''
    Helper that takes in an np array (generally raw strings from the CSV) and
    converts it to the specified type or None via convert_fn
    '''
    converted = np.array([
        convert_fn(val)
        for val in col
    ])
    are_nones = pd.isnull(converted)
    return converted, are_nones


def _clean_continuous__inplace(col: np.array, convert_fn) -> Tuple[np.array, int]:
    '''
    Helper that takes in an np array (generally raw strings from the CSV) and
    cleans it by drawing values for empty/problematic cells from a normal distribution
    formed by the other cells' value
    '''
    converted, are_nones = _convert(col, convert_fn)
    return converted, are_nones.sum()

######################## CATEGORICAL ########################

def _convert_int_or_null(val: str) -> int | None:
    if val == None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def clean_categorical__inplace(col: np.array) -> Tuple[np.array, int]:
    converted, are_nones = _convert(col, _convert_int_or_null)
    return converted, are_nones.sum()

######################## NUMERIC ########################

def _convert_float_or_null(val: str) -> float | None:
    try:
        return float(val)
    except Exception:
        return None

def clean_numerical__inplace(col: np.array) -> Tuple[np.array, int]:
    return _clean_continuous__inplace(col, _convert_float_or_null)


######################## DATE OR TIME ########################

def _convert_to_datetimems_or_none(val: str) -> float | None:
    # dataset had whitespace... argh    
    if val == None:
        return None
    val = val.strip()

    # is a time H:M:S,junk
    if ":" in val:
        # only keep 3 time parts HMS
        els = val.split(':')[:3]
    
        def _convert_to_seconds(accum: int, cur: str):
            # sometimes had other data after seconds, not necessary, only keep two digits 
            return (accum*60)+int(cur[:2])
        
        return reduce(_convert_to_seconds, els, 1/60)

    # is a date
    for format in DATE_FORMATS:
        try:
            return datetime.strptime(val, format).timestamp()
        except Exception:
            pass

    return None

def clean_datetime__inplace(col: np.array) -> Tuple[np.array, int]:
    return _clean_continuous__inplace(col, _convert_to_datetimems_or_none)


######################## CHARACTER ########################

def _convert_to_str_or_none(val: str) -> str | None:
    return None if (val in NULL_STRINGS or val is None) else val


def clean_character__inplace(col: np.array) -> Tuple[np.array, int]:
    converted, are_nones = _convert(col, _convert_to_str_or_none)
    return converted, are_nones.sum()


######################## RESPONSE TO OPEN QUESTION ########################

def clean_response_to_open_ended__inplace(col: np.array) -> Tuple[np.array, int]:
    return clean_character__inplace(col)


######################## CLEANER ########################

def clean_data__inplace(
        mtx: np.array,
        ordered_col_metadata: list,
        num_exclude_threshold: int,
    ) -> Tuple[np.array, list[str]]:
    trans = mtx.transpose()
    passed_threshold_state = []
    passed_threshold_col_names = []
    cleaned_cols = np.array([])
    for col, (col_name, col_type, cleaner) in zip(trans, ordered_col_metadata):
        try:
            converted, num_nones = cleaner(col)
            passed = num_nones >= num_exclude_threshold
            passed_threshold_state.append(passed)
            if passed:
                cleaned_cols.append(converted)
                passed_threshold_col_names.append(col_name)
        except Exception as e:
            print(f"There was an error processing column '{col_name}' of type '{col_type}'")
            raise e
    return cleaned_cols, passed_threshold_col_names

######################## ENTRYPOINT ########################

def first_pass(training_data_df: pd.DataFrame, codebook_df: pd.DataFrame, outcome_df: pd.DataFrame, none_threshold: int) -> np.array:

    # mapping each column to the cleaning function necessary for its type
    COLTYPE_TO_CLEANER = {
        'character [almost exclusively empty strings]': clean_character__inplace,
        'numeric': clean_numerical__inplace,
        'categorical': clean_categorical__inplace,
        'response to open-ended question': clean_response_to_open_ended__inplace,
        'date or time': clean_datetime__inplace,   
    }

    colname_to_type = {
        r["var_name"]: r["type_var"]
        for _, r in codebook_df.iterrows()
    }

    # Optionally, go through and remove the rows that don't have an outcome. This makes follow-up passes a lot easier
    present_df = outcome_df[(outcome_df['new_child'] == 0) | (outcome_df['new_child'] == 1)]
    present_ids = set(present_df['nomem_encr'])

    print("Pruning - converting type")
    training_data_df['nomem_encr'] = training_data_df['nomem_encr'].astype(int)
    print("Pruning - filtering present outcomes")
    pruned_df = training_data_df[training_data_df['nomem_encr'].isin(present_ids)].copy()

    # Get rid of metadata/extra data we don't need anymore
    del training_data_df
    del outcome_df
    del present_df

    print("=== Before Cleaning ===")
    print(pruned_df)

    remove_cols = set()

    desired_types = {
        'numeric': float,
        'categorical': str,
        'date or time': str
    }

    for col in tqdm(pruned_df.columns):
        col_type = colname_to_type[col]
        if col_type == "numeric":
            # Convert all numbers to floats because it doesn't seem possible to differentiate and we'll lose all the floats otherwise
            pruned_df[col] = pruned_df[col].apply(lambda x: _convert_float_or_null(x))
        elif col_type == "date or time":
            # Convert all dates to seconds (floats)
            pruned_df[col] = pruned_df[col].apply(lambda x: _convert_to_datetimems_or_none(x))
        elif col_type == "categorical":
            # Convert all categories to ints (or nones)
            pruned_df[col] = pruned_df[col].apply(lambda x: _convert_int_or_null(x))
        elif col_type not in desired_types:
            # Turns out that dropping is stupid slow. Just track and subselect later
            remove_cols.add(col)


    print("=== After Cleaning ===")
    print(pruned_df)
    keep_cols = set(pruned_df.columns) - remove_cols

    cleaned_df = pruned_df[list(keep_cols)].copy()
    print("=== After Removing Columns ===")
    # Fill all na values with nan b/c fuck it
    cleaned_df.fillna(value=np.nan, inplace=True)
    print(cleaned_df)

    return cleaned_df
    
def second_pass(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: remove all the columns where there's a very high percentage of nan values.
    targets = 0
    target_cols = set(df.columns)
    # A threshold of 0.7 removes 21k columns, 0.8 17k, 0.9 12k, 1.0 3k
    NA_THRESHOLD = 0.8
    for col in df.columns:
        na_count = df[col].isna().sum()
        na_pct = na_count/len(df[col])
        if na_pct >= NA_THRESHOLD:
            targets += 1
            target_cols.remove(col)

    print(f'Targets for removal: {targets}')
    useful_df = df[list(target_cols)].copy()

    print("=== Before Imputation ===")
    print(useful_df)

    imputer = KNNImputer(n_neighbors=5)
    # This takes a long time and I don't think there's a good way to keep track of it

    print("Sanity check:", "nomem_encr" in set(useful_df.columns))
    useful_df[:] = imputer.fit_transform(useful_df)
    print("=== After Imputation ===")
    # This creates a np array. Turn it back into a dataframe
    print("Sanity check:", "nomem_encr" in set(useful_df.columns))
    useful_df.set_index("nomem_encr", inplace=True)
    print(useful_df)
    useful_df.to_csv("imputed.csv", header=True)


def load_raw_df(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        # This should do basically all the cleaning we need. Just need to convert the columns to the right types afterwards
        return pd.read_csv(path, header=0, dtype=str, na_filter=True, na_values=NULL_STRINGS)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Invalid file format")

def load_dataframe(path: str, low_memory=False) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, header=0, low_memory=low_memory)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Invalid file format")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", help="path to the codebook file")
    parser.add_argument("--training", help="path to the training data file", default="", required=False)
    parser.add_argument("--outcome", help="path to the outcome data file", default="", required=True)
    parser.add_argument("--cleaned", help="path to the cleaned data file", default="", required=False)
    parser.add_argument("--threshold", type=int, help="threshold", default=0)
    
    args = parser.parse_args()
    cleaned_df = None
    
    print("Loading codebook")
    codebook_df = load_dataframe(args.codebook)
    outcome_df = pd.read_csv(args.outcome, header=0, na_filter=True, na_values=NULL_STRINGS)
    if args.cleaned == "":
        # Load the raw data
        print("Loading raw data")
        raw_df = load_raw_df(args.training)
        if args.training.endswith(".csv"):
            raw_df.to_parquet('raw.parquet')
        # Load the codebook
        print("Cleaning dataframe - first path")
        cleaned_df = first_pass(raw_df, codebook_df, outcome_df, args.threshold)
        print("Finished first path cleaning")
        cleaned_df.to_parquet('cleaned.parquet')
        print("Saved cleaned version")
    else:
        print("Loading cleaned dataframe")
        cleaned_df = load_dataframe(args.cleaned)

    # Do the imputation
    print("Part 2 - Imputation")
    print(cleaned_df)
    second_pass(cleaned_df)
    