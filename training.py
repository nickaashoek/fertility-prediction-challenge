"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

######################## GLOBALS ########################
DEVICE = "cpu"
DEFAULT_SGE_KWARGS = {
    "lr": 0.0001,
    "momentum": 0.9,
    "nesterov": False,
}
DEFAULT_BATCH_SIZE = 20
DEFAULT_EPOCHS = 10
DEFAULT_NODES_PER_LAYER = (100, 50, 25, 10)
USE_SAMPLER = False

######################## TrainingDataset ########################
def normalize_col(col, df: pd.DataFrame):
    col_max, col_min = df[col].max(), df[col].min()
    if col_max == col_min:
        # If all the values are the same, set to 0
        df[col] = df[col].apply(lambda x: 0)
    else:
        df[col] =  df[col].apply(lambda x: (x-col_min)/(col_max-col_min))

class ClassifierPreFerDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, outcome_df: pd.DataFrame):
        print("=== Setting up dataset ===")
        # First, drop all the columns that don't have an outcome because they don't help with training
        data_df = data_df[data_df['outcome_available'] == 1.0]
        print("Input frame has:", len(data_df.columns), "cols")
        joined_df = data_df.merge(outcome_df, on='nomem_encr')
        # Drop the columns we don't need anymore, leaving only the features and the outcome
        joined_df.drop(columns=['nomem_encr'], inplace=True)
        joined_df.drop(columns=['outcome_available'], inplace=True)
        print("=== Remove all NaNs ===")
        joined_df.fillna(0, inplace=True)
        for col in joined_df.columns:
            normalize_col(col, joined_df)
        self.data = torch.from_numpy(joined_df.to_numpy()).type(torch.float32)
        # -1 because the outcome doesn't count as a feature
        self.num_features = len(joined_df.columns) - 1

        assert len(self.data), "training data empty"
        # Going to do explicit matching, so the below isn't necessary
        # assert len(self.features_data) == len(self.outcome_data), f"training data had {len(self.features_data)} rows which did not match outcome data which had {len(self.outcome_data)} rows"

        uniques, counts = joined_df['new_child'].unique(), joined_df['new_child'].value_counts()
        self.num_labels = len(uniques)
        # below is used for trying to even out the fact that there are more 0s than anything else
        print("=== Setting Sampler ===")
        self.sampler = self._set_sampler(uniques, counts)
    
    def _set_sampler(self, uniques: np.array, counts: np.array) -> None:
        self.sampler = None
        if not USE_SAMPLER:
            return
        
        counts_lookup = {
            v: c
            for v, c in zip(uniques, counts)
        }
        weights = torch.Tensor(
            [
                counts_lookup[v.item()]
                for i, v in enumerate(torch.flatten(self.outcome_data))
            ]
        )
        self.sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        # useful for debugging
        # val = int(random.random() < 0.75)
        #return torch.Tensor([np.float32(1)] * self.num_features), np.float32(val)
        return self.data[i][:-1], int(self.data[i][-1])


######################## ClassifierNeuralNetwork ########################

class ClassifierNeuralNetwork(nn.Module):
    def __init__(self, num_features: int, nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER, num_labels: int = None):
        super().__init__()
        assert num_features > 0, "You cannot have no features (cols) in your dataset"
        assert nodes_per_layer, "You must specify the layers in your network"

        # first layer should be size of features
        if nodes_per_layer[0] != num_features:
            nodes_per_layer = [num_features] + [*nodes_per_layer]
        # must end with one layer that has a neuron for each label
        if nodes_per_layer[-1] != num_labels:
            nodes_per_layer = [*nodes_per_layer] + [num_labels]

        unflattened_args = [
            (
                nn.Linear(nodes_per_layer[i], nodes_per_layer[i+1]),
                nn.ReLU()
            )
            for i in range(len(nodes_per_layer)-1) # do not want to do this for the last value
        ]
        # flatten
        args = [
            entity
            for entity_pair in unflattened_args
            for entity in entity_pair
        ]
        # do not end on ReLU, instead need output neuron
        args.pop()
        self.linear_relu_stack = nn.Sequential(*args)

    def forward(self, batch: torch.Tensor):
        logits = self.linear_relu_stack(batch)
        return logits
    
    @staticmethod
    def train_loop(
        model: ClassifierNeuralNetwork,
        dataloader: DataLoader,
        optimizer,
        loss_fn
    ):
        # likely do not need, but just in case - this moves the same model instance to the GPU
        model.to(DEVICE)
        model.train()
        for i_batch, (features, outcomes) in enumerate(dataloader):
            outcomes = outcomes.type(torch.LongTensor)
            optimizer.zero_grad()
            # OG tensor isntance still on CPU - only returned goes to GPU
            pred = model(features.to(DEVICE))
            loss = loss_fn(pred, outcomes.to(DEVICE))

            # backpropogate
            loss.backward()
            optimizer.step()
        
        return model

    @classmethod
    def create_and_train(
        cls,
        training_df: pd.DataFrame, 
        outcome_df: pd.DataFrame,
        get_optimizer,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loss_fn = None,
        epochs: int = DEFAULT_EPOCHS,
    ) -> ClassifierNeuralNetwork:

        dataset = ClassifierPreFerDataset(training_df, outcome_df)
        # you are only allowed to shuffle if you do not have a sampler
        shuffle = not dataset.sampler
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=dataset.sampler)

        # OG model goes to GPU as well
        model = cls(dataset.num_features, nodes_per_layer, num_labels=dataset.num_labels).to(DEVICE)
        optimizer = get_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        for _ in range(epochs):
            # MSELoss as default since regression task
            cls.train_loop(model, dataloader, optimizer, loss_fn or nn.CrossEntropyLoss())
            scheduler.step()

        # cls.test_model(model, dataloader, nn.CrossEntropyLoss())

        return model
    
    @classmethod
    def create_and_train_SGE(
        cls,
        training_df: pd.DataFrame, 
        outcome_df: pd.DataFrame,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loss_fn = None,
        epochs: int = DEFAULT_EPOCHS,
        **SGE_kwargs,
    ) -> ClassifierNeuralNetwork:
        total_kwargs = {
            # good initial values should no other args be passed
            **DEFAULT_SGE_KWARGS,
            # upsert the passed in values
            **SGE_kwargs
        }
        return cls.create_and_train(
            training_df,
            outcome_df,
            get_optimizer=lambda m: torch.optim.SGD(m.parameters(), **total_kwargs),
            nodes_per_layer=nodes_per_layer,
            batch_size=batch_size,
            loss_fn=loss_fn,
            epochs=epochs,
        )

    @staticmethod
    def test_model(
        model: ClassifierNeuralNetwork,
        dataloader: DataLoader,
        loss_fn = None
    ) -> tuple[float, float]:
        
        if not loss_fn:
            loss_fn = nn.CrossEntropyLoss()
        
        # likely do not need, but just in case - this moves the same model instance to the GPU
        model.to(DEVICE)
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        correct, test_loss = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for i, (features, outcome) in enumerate(dataloader):
                # OG tensor isntance still on CPU - only returned goes to GPU
                pred = model(features.to(DEVICE))
                dev_outcome = outcome.to(DEVICE)
                # print(pred.argmax(1))
                # print(dev_outcome)
                # print('\n\n')
                if i % 100 == 0:
                    print(pred)
                correct += (pred.argmax(1) == dev_outcome).type(torch.float).sum().item()
                test_loss += loss_fn(pred, dev_outcome).item()

        correct /= size
        test_loss /= num_batches
        print(f"Correct Ratio: {correct} Avg loss: {test_loss:>8f} \n")
        return correct, test_loss

    def predict(
        self,
        predict_df: pd.DataFrame,
    ) -> tuple[float, float]:
        print("=== Making Predictions ===")
        predictions = []
        with torch.no_grad():
            for _, row in predict_df.iterrows():
                feature = row.to_numpy()
                feature_tensor = torch.from_numpy(feature)
                pred = self(feature_tensor.to(DEVICE))
                predictions.append(pred.argmax().item())
        return predictions
    
def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    model = ClassifierNeuralNetwork.create_and_train_SGE(
        cleaned_df,
        outcome_df,
        DEFAULT_NODES_PER_LAYER,
        DEFAULT_BATCH_SIZE,
        nn.CrossEntropyLoss(),
        DEFAULT_EPOCHS,
        **DEFAULT_SGE_KWARGS,
    )
    print("=== Done training model ===")
    torch.save(model.state_dict(), "model.joblib")
    print("=== Model Saved ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", help="path to the training data file", default="", required=False)
    parser.add_argument("--outcome", help="path to the outcome data file", default="", required=False)
    args = parser.parse_args()

    outcome_df = pd.read_csv(args.outcome)
    print("=== Loaded outcome df ===")
    train_df = pd.read_csv(args.training, low_memory=False, index_col=0)
    print("=== Loaded training df ===")
    print("=== Training and saving model ===")
    train_save_model(train_df, outcome_df)