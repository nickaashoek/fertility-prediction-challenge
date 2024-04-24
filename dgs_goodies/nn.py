from __future__ import annotations

import numpy as np
import pandas as pd
from ray import tune
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


######################## GLOBALS ########################

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEFAULT_NODES_PER_LAYER = (10000, 5000, 1000, 500, 50, 10)

######################## TrainingDataset ########################

class PreFerDataset(Dataset):
    def __init__(self, features_data_filepath: str, outcome_data_filepath: str):
        self.features_data = pd.read_csv(features_data_filepath, header=0).to_numpy()
        self.outcome_data = pd.read_csv(outcome_data_filepath, header=0).to_numpy()
        assert len(self.features_data), "training data empty"
        assert len(self.features_data) == len(self.outcome_data), f"training data had {len(self.features_data)} rows which did not match outcome data which had {len(self.outcome_data)} rows"

    def __len__(self):
        return len(self.features_data)

    def __getitem__(self, i: int | np.array[int]) -> tuple[np.array, int] | tuple[np.array, np.array]:
        return self.features_data[i], self.outcome_data[i]


######################## NeuralNetwork ########################

class NeuralNetwork(nn.Module):
    def __init__(self, num_features: int, nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER):
        super().__init__()
        assert num_features > 0, "You cannot have no features (cols) in your dataset"
        assert nodes_per_layer, "You must specify the layers in your network"

        # first layer should be size of features
        nodes_per_layer = [num_features] + [*nodes_per_layer]
        # must end with one layer which spits out the anser
        if nodes_per_layer[-1] != 1:
            nodes_per_layer = [*nodes_per_layer] + [1]

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
        self.linear_relu_stack = nn.Sequential(*args)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    @staticmethod
    def train_loop(
        model: NeuralNetwork,
        dataloader: DataLoader,
        optimizer,
        loss_fn
    ):
        optimizer.zero_grad()
        model.train()
        for i_batch, (features, outcomes) in enumerate(dataloader):
            pred = model(features)
            loss = loss_fn(pred, outcomes)

            # backpropogate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        return model

    @classmethod
    def create_and_train(
        cls,
        training_data_filepath: str, 
        outcome_data_filepath: str,
        get_optimizer,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = 500,
        loss_fn = None,
        epochs: int = 20,
    ) -> NeuralNetwork:

        dataset = PreFerDataset(training_data_filepath, outcome_data_filepath)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = cls(len(dataset), nodes_per_layer).to(DEVICE)
        optimizer = get_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ExponentialLR

        for _ in range(epochs):
            # MSELoss as default since regression task
            cls.train_loop(model, dataloader, optimizer, loss_fn or nn.MSELoss())
            scheduler.step()

        return model
    
    @classmethod
    def create_and_train_SGE(
        cls,
        training_data_filepath: str, 
        outcome_data_filepath: str,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = 500,
        loss_fn = None,
        epochs: int = 20,
        **SGE_kwargs,
    ) -> NeuralNetwork:
        total_kwargs = {
            # good initial values should no other args be passed
            "lr": 0.001,
            "momentum": 0.9,
            "nesterov": True,
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
        model: NeuralNetwork,
        dataloader: DataLoader,
        loss_fn = None
    ) -> tuple[float, float]:
        
        if not loss_fn:
            loss_fn = nn.MSELoss()
        
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for features, outcomes in dataloader:
                pred = model(features)
                test_loss += loss_fn(pred, outcomes).item()
                correct += (pred.argmax(1) == outcomes).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, correct
    
    @classmethod
    def train_and_test_SGE(
        cls,
        training_data_filepath: str, 
        outcome_data_filepath: str,
        testing_features_filepath: str,
        testing_outcomes_filepath: str,
        nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER,
        batch_size: int = 500,
        loss_fn = None,
        epochs: int = 20,
        **SGE_kwargs,
    ) -> tuple[NeuralNetwork, float, float]:
        
        test_dataset = PreFerDataset(testing_features_filepath, testing_outcomes_filepath)
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
        test_loss, correct = cls.test_model(model, test_dataloader, loss_fn)
        return model, test_loss, correct


######################## HYPERPARAM OPT ########################

def meow():
    pass


######################## ENTRYPOINT ########################

# I think that this should write the model itself to somewhere, no? so no return? need to think
def get_best_model_entrypoint(
        cleaned_data_filepath: str,
        outcomes_filepath: str,
        testing_features_filepath: str,
        testing_outcomes_filepath: str,
    ) -> None:
    # just doing this for now - should do hyperparam tuning
    model, test_loss, correct = NeuralNetwork.train_and_test_SGE(
        cleaned_data_filepath,
        outcomes_filepath,
        testing_features_filepath,
        testing_outcomes_filepath,
    )


# should make this a CLI
if __name__ == "__main__":
    cleaned_data_filepath, outcomes_filepath, testing_features_filepath, testing_outcomes_filepath = sys.argv[1:]
    best_model = get_best_model_entrypoint(cleaned_data_filepath, outcomes_filepath, testing_features_filepath, testing_outcomes_filepath)
