from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import random
import yaml

######################## GLOBALS ########################

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEFAULT_SGE_KWARGS = {
    "lr": 0.001,
    "momentum": 0.9,
    "nesterov": True,
}
DEFAULT_BATCH_SIZE = 25
DEFAULT_EPOCHS = 100
DEFAULT_NODES_PER_LAYER = (100, 50, 25, 10)
USE_SAMPLER = False

######################## TrainingDataset ########################

class ClassifierPreFerDataset(Dataset):
    def __init__(self, features_data_filepath: str, outcome_data_filepath: str):
        # must be float32 to use apple GPU
        data_df = pd.read_csv(features_data_filepath, header=0, dtype=np.float32)
        outcome_df = pd.read_csv(outcome_data_filepath, header=0, dtype=np.float32)
        # A useful sanity check
        for val in data_df['nomem_encr']:
            match = len(outcome_df[outcome_df['nomem_encr'] == val])
            if match == 0:
                print(f"Could not find a match for {val}")
        
        joined_df = data_df.merge(outcome_df, on='nomem_encr')
        self.data = torch.from_numpy(joined_df.to_numpy())
        # -1 because the nomem_encr doesn't count as a feature
        self.num_features = len(data_df.columns) - 1
        
        
        self.features_data = torch.from_numpy(pd.read_csv(features_data_filepath, header=0, dtype=np.float32).to_numpy())
        _, self.num_features = self.features_data.shape

        self.outcome_data = torch.from_numpy(pd.read_csv(outcome_data_filepath, header=0, dtype=np.float32).to_numpy())

        assert len(self.features_data), "training data empty"
        # Going to do explicit matching, so the below isn't necessary
        # assert len(self.features_data) == len(self.outcome_data), f"training data had {len(self.features_data)} rows which did not match outcome data which had {len(self.outcome_data)} rows"

        uniques, counts = np.unique(self.outcome_data, return_counts=True)
        self.num_labels = len(uniques)
        # below is used for trying to even out the fact that there are more 0s than anything else
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
        return len(self.features_data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        # useful for debugging
        # val = int(random.random() < 0.75)
        #return torch.Tensor([np.float32(1)] * self.num_features), np.float32(val)
        return self.features_data[i], self.outcome_data[i][0]


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
        training_data_filepath: str, 
        outcome_data_filepath: str,
        get_optimizer,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loss_fn = None,
        epochs: int = DEFAULT_EPOCHS,
    ) -> ClassifierNeuralNetwork:

        dataset = ClassifierPreFerDataset(training_data_filepath, outcome_data_filepath)
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

        return model
    
    @classmethod
    def create_and_train_SGE(
        cls,
        training_data_filepath: str, 
        outcome_data_filepath: str,
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
            training_data_filepath,
            outcome_data_filepath,
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
            for features, outcome in dataloader:
                # OG tensor isntance still on CPU - only returned goes to GPU
                pred = model(features.to(DEVICE))
                dev_outcome = outcome.to(DEVICE)
                # print(pred.argmax(1))
                # print(dev_outcome)
                # print('\n\n')
                correct += (pred.argmax(1) == dev_outcome).type(torch.float).sum().item()
                test_loss += loss_fn(pred, dev_outcome).item()

        correct /= size
        test_loss /= num_batches
        print(f"Correct Ratio: {correct} Avg loss: {test_loss:>8f} \n")
        return correct, test_loss
    
    @classmethod
    def train_and_test_SGE(
        cls,
        training_data_filepath: str, 
        outcome_data_filepath: str,
        testing_features_filepath: str,
        testing_outcomes_filepath: str,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loss_fn = None,
        epochs: int = DEFAULT_EPOCHS,
        **SGE_kwargs,
    ) -> tuple[ClassifierNeuralNetwork, float, float]:
        
        test_dataset = ClassifierPreFerDataset(testing_features_filepath, testing_outcomes_filepath)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model = cls.create_and_train_SGE(
            training_data_filepath,
            outcome_data_filepath,
            nodes_per_layer,
            batch_size,
            loss_fn,
            epochs,
            **SGE_kwargs
        )
        correct, test_loss = cls.test_model(model, test_dataloader, loss_fn)
        return model, correct, test_loss


######################## ENTRYPOINT ########################

# I think that this should write the model itself to somewhere, no? so no return? need to think
def get_classifer_entrypoint(
        cleaned_data_filepath: str,
        outcomes_filepath: str,
        testing_features_filepath: str,
        testing_outcomes_filepath: str,
        params_filepath: str,
    ) -> None:
    params_file = open(params_filepath, "r")
    params = yaml.safe_load(params_file)
    print(params)
    params_file.close()
    # exit(1)
    # just doing this for now - should do hyperparam tuning
    model, correct, test_loss = ClassifierNeuralNetwork.train_and_test_SGE(
        cleaned_data_filepath,
        outcomes_filepath,
        testing_features_filepath,
        testing_outcomes_filepath,
        # ALL BELOW SHOULD BE CLI INPUTS
        nodes_per_layer=params['nodes_per_layer'],
        batch_size=params['batch_size'],
        loss_fn=nn.CrossEntropyLoss(),
        epochs=params['epochs'],
        # SGE args
        lr=params['lr'],
        momentum=params['momentum'],
        nesterov=params['nesterov'],
    )
    print("After running the model", correct, test_loss)


# should make this a CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", help="path to the cleaned data file") 
    parser.add_argument("--outcomes", help="path to the outcomes file")
    parser.add_argument("--testing_features", help="path to the testing features file")
    parser.add_argument("--testing_outcomes", help="path to the testing outcomes file")
    parser.add_argument("--params", help="path to the hyperparameters file")
    
    args = parser.parse_args()
    
    best_model = get_classifer_entrypoint(
        args.cleaned,
        args.outcomes,
        args.testing_features,
        args.testing_outcomes,
        args.params
    ) 