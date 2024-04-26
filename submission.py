"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
import joblib
import datetime
from functools import reduce
import numpy as np

DATE_FORMATS = (
    '%d.%m.%Y',
    '%d-%m-%Y',
    '%d/%m/%Y',
)

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

def first_pass(training_data_df: pd.DataFrame) -> np.array:

    # mapping each column to the cleaning function necessary for its type

    print("=== Before Cleaning ===")
    print(training_data_df)

    remove_cols = set()
    dtypes = training_data_df.dtypes
    for col in training_data_df.columns:
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
    for col in training_data_df.columns:
        val_count = training_data_df[col].value_counts()
        if len(val_count) == 0 or len(val_count) == 1:
            # These are objectively worthless
            remove_cols.add(col)
    
    print(f'Remove {len(remove_cols)} columns (single value)')
    keep_cols = set(training_data_df.columns) - remove_cols
    cleaned_df = training_data_df[list(keep_cols)].copy()
    print("=== After Cleaning ===")


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

    imputer = KNNImputer(n_neighbors=5)
    # This takes a long time and I don't think there's a good way to keep track of it
    print("=== Imputing on to_impute ===")
    to_impute[:] = imputer.fit_transform(to_impute)
    print("=== Finished Imputing ===")
    # Merge the imputed rows back into the original dataframe
    print("=== Starting Merge ===")
    cols = list(set(to_impute.columns) - set(["nomem_encr", "outcome_available"]))
    useful_df.loc[useful_df["nomem_encr"].isin(to_impute["nomem_encr"]), cols] = to_impute[cols]
    print("=== Merge finished ===")
    return useful_df


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## This script contains a bare minimum working example
    # Create new variable with age
    cleaned_df = first_pass(df)
    imputed_df = second_pass(cleaned_df)
    return imputed_df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
