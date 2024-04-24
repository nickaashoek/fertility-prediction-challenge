from __future__ import annotations

import numpy as np
import pandas as pd
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
DEFAULT_SGE_KWARGS = {
    "lr": 0.001,
    "momentum": 0.9,
    "nesterov": True,
}
DEFAULT_BATCH_SIZE = 25
DEFAULT_EPOCHS = 100
DEFAULT_NODES_PER_LAYER = (100, 50, 25, 10)
DEBUG = False

######################## TrainingDataset ########################

class RegressionPreFerDataset(Dataset):
    def __init__(self, features_data_filepath: str, outcome_data_filepath: str):
        # must be float32 to use apple GPU
        self.features_data = torch.from_numpy(pd.read_csv(features_data_filepath, header=0, dtype=np.float32).to_numpy())
        _, self.num_features = self.features_data.shape

        self.outcome_data = torch.from_numpy(pd.read_csv(outcome_data_filepath, header=0, dtype=np.float32).to_numpy())
        assert len(self.features_data), "training data empty"
        assert len(self.features_data) == len(self.outcome_data), f"training data had {len(self.features_data)} rows which did not match outcome data which had {len(self.outcome_data)} rows"
    
    def __len__(self):
        return len(self.features_data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # uncomment for testing - this shows that the model architecture itself is able to converge on simpler data
        # return torch.from_numpy(np.array([np.float32(1.0)]*self.num_features)), torch.from_numpy(np.array([np.float32(1.0)]))
        return self.features_data[i], self.outcome_data[i]


######################## RegressionNeuralNetwork ########################

class RegressionNeuralNetwork(nn.Module):
    def __init__(self, num_features: int, nodes_per_layer: tuple[int, ...] = DEFAULT_NODES_PER_LAYER):
        super().__init__()
        assert num_features > 0, "You cannot have no features (cols) in your dataset"
        assert nodes_per_layer, "You must specify the layers in your network"

        # first layer should be size of features
        if nodes_per_layer[0] != num_features:
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
        model: RegressionNeuralNetwork,
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
    ) -> RegressionNeuralNetwork:

        dataset = RegressionPreFerDataset(training_data_filepath, outcome_data_filepath)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # OG model goes to GPU as well
        model = cls(dataset.num_features, nodes_per_layer).to(DEVICE)
        optimizer = get_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
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
        batch_size: int = DEFAULT_BATCH_SIZE,
        loss_fn = None,
        epochs: int = DEFAULT_EPOCHS,
        **SGE_kwargs,
    ) -> RegressionNeuralNetwork:
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
        model: RegressionNeuralNetwork,
        dataloader: DataLoader,
        loss_fn = None
    ) -> tuple[float, float]:
        
        if not loss_fn:
            loss_fn = nn.MSELoss()
        
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

                # helpful for debugging/checking
                if DEBUG:
                    predval = torch.round(pred).item()
                    print(pred)
                    print(predval)
                    if predval:
                        print('pred was nonzero')
                        #raise Exception
                    print(dev_outcome.item())
                    print('\n\n')
    
                # correct += int(pred.item() == dev_outcome.item())
                test_loss += loss_fn(pred, dev_outcome).item()

        test_loss /= num_batches
        # correct /=  size
        # print(f"Correct Ratio: {correct} Avg loss: {test_loss:>8f} \n")
        print(f"Avg loss: {test_loss:>8f} \n")
        return test_loss
    
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
    ) -> tuple[RegressionNeuralNetwork, float, float]:
        
        test_dataset = RegressionPreFerDataset(testing_features_filepath, testing_outcomes_filepath)
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
        test_loss = cls.test_model(model, test_dataloader, loss_fn)
        return model, test_loss


######################## ENTRYPOINT ########################

# I think that this should write the model itself to somewhere, no? so no return? need to think
def get_model_entrypoint(
        cleaned_data_filepath: str,
        outcomes_filepath: str,
        testing_features_filepath: str,
        testing_outcomes_filepath: str,
    ) -> None:
    # just doing this for now - should do hyperparam tuning
    model, test_loss = RegressionNeuralNetwork.train_and_test_SGE(
        cleaned_data_filepath,
        outcomes_filepath,
        testing_features_filepath,
        testing_outcomes_filepath,
        # ALL BELOW SHOULD BE CLI INPUTS
        nodes_per_layer=DEFAULT_NODES_PER_LAYER,
        batch_size=20,
        loss_fn=nn.MSELoss(),
        epochs=1000,
        # SGE args
        lr=0.0001,
        momentum=0.9,
        nesterov=False,
    )


# should make this a CLI
if __name__ == "__main__":
    cleaned_data_filepath, outcomes_filepath, testing_features_filepath, testing_outcomes_filepath = sys.argv[1:]
    best_model = get_model_entrypoint(cleaned_data_filepath, outcomes_filepath, testing_features_filepath, testing_outcomes_filepath)
