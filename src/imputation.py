from datetime import datetime
from functools import reduce
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
import os
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

def _convert_to_datetimems_or_none(val: str):
    # dataset had whitespace... argh    
    if val == None or type(val) == float:
        return np.nan
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

    return np.nan

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

def first_pass(training_data_df: pd.DataFrame) -> np.array:

    # mapping each column to the cleaning function necessary for its type

    print("=== Before Cleaning ===")
    print(training_data_df)

    remove_cols = set()
    dtypes = training_data_df.dtypes
    for col in tqdm(training_data_df.columns):
        col_type = dtypes[col]
        if col == "nomem_encr":
            # Don't touch the Id column
            continue
        if pd.api.types.is_integer_dtype(col_type):
            # Don't bother converting something that's already an integer => These are numerics
            continue
        elif pd.api.types.is_float_dtype(col_type):
            # Don't bother converting floats beause they're already set => These are categories
            continue
        else:
            # This could be a date for all we know, but actually parsing it is a massive pain in the ass
            # Just guess and check later => whatever's left after cleaning will be a float
            def try_datetime(val):
                try:
                    converted = _convert_to_datetimems_or_none(val)
                    return converted
                except:
                    return np.nan
            training_data_df[col] = training_data_df[col].apply(lambda x: try_datetime(x))

    # Go through and remove columns that only have one value
    training_data_df.fillna(value=np.nan, inplace=True)
    print("=== After converting ===")
    print(training_data_df)
    for col in tqdm(training_data_df.columns):
        val_count = training_data_df[col].value_counts()
        if len(val_count) == 0 or len(val_count) == 1:
            # These are objectively worthless
            remove_cols.add(col)
    
    print(f'Remove {len(remove_cols)} columns (single value)')
    keep_cols = set(training_data_df.columns) - remove_cols
    cleaned_df = training_data_df[list(keep_cols)].copy()
    print("=== After Cleaning ===")
    print(cleaned_df)


    return cleaned_df
    
def second_pass(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: remove all the columns where there's a very high percentage of nan values so that the imputation is a bit faster.
    print("=== Removing high nan rows ===")
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

    print(f'{targets} columsn to be removed via NA_Threshold')
    useful_df = df[list(target_cols)].copy()

    print("=== Before Imputation ===")

    """
    Get a useful dataframe that only has outcomes, perform imputation on that dataframe for speed reasons
    Then, replace the rows in the original dataframe with the imputed values
    """

    to_impute = useful_df[useful_df['outcome_available'] == 1].copy()
    print(to_impute)

    imputer = KNNImputer(n_neighbors=5)
    # This takes a long time and I don't think there's a good way to keep track of it
    print("=== Imputing on to_impute ===")
    if not os.path.exists("imputed_base.csv"):
        to_impute[:] = imputer.fit_transform(to_impute)
        to_impute.to_csv("imputed_base.csv")
    else:
        to_impute = pd.read_csv("imputed_base.csv", low_memory=False)
    print("=== Finished Imputing ===")
    print(to_impute)
    # Merge the imputed rows back into the original dataframe
    print("=== Starting Merge ===")
    print(to_impute["outcome_available"].value_counts())
    cols = list(set(to_impute.columns) - set(["nomem_encr", "outcome_available"]))
    useful_df.loc[useful_df["nomem_encr"].isin(to_impute["nomem_encr"]), cols] = to_impute[cols]
    print(useful_df["outcome_available"].value_counts())
    print("=== Merge finished ===")
    return useful_df


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
    parser.add_argument("--training", help="path to the training data file", default="", required=False)
    parser.add_argument("--cleaned", help="path to the cleaned data file", default="", required=False)

    args = parser.parse_args()

    print("=== Loading dataframe ===")
    cleaned_v1 = None
    if args.training:
        training_data_df = pd.read_csv(args.training, low_memory=False)
        print("=== Loaded dataframe ===")
        cleaned_v1 = first_pass(training_data_df)
        cleaned_v1.to_csv("cleaned_v1.csv")
    else:
        cleaned_v1 = pd.read_csv("cleaned_v1.csv", low_memory=False)
    imputed = second_pass(cleaned_v1)
    imputed.to_csv("imputed.csv")