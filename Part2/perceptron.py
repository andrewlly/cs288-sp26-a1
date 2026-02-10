"""Perceptron model model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from features import make_featurize
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


@dataclass(frozen=True)
class DataPointWithFeatures(DataPoint):
    features: Dict[str, float]


def featurize_data(
    data: List[DataPoint], feature_types: Set[str]
) -> List[DataPointWithFeatures]:
    """Add features to each datapoint based on feature types"""
    # TODO: Implement this!
    featurizer = make_featurize(feature_types)
    ans = []
    
    for dp in data:
        f = featurizer(dp.text)
        new_dp = DataPointWithFeatures(dp.id, dp.text, dp.label, f)
        ans.append(new_dp)
        
    return ans


class PerceptronModel:
    """Perceptron model for classification."""

    def __init__(self):
        self.weights: Dict[str, float] = defaultdict(float)
        self.labels: Set[str] = set()

    def _get_weight_key(self, feature: str, label: str) -> str:
        """An internal hash function to build keys of self.weights (needed for tests)"""
        return feature + "#" + str(label)

    def score(self, datapoint: DataPointWithFeatures, label: str) -> float:
        """Compute the score of a class given the input.

        Inputs:
            datapoint (Datapoint): a single datapoint with features populated
            label (str): label

        Returns:
            The output score.
        """
        # TODO: Implement this! Expected # of lines: <10
        score_val = 0.0
        for feat_name, feat_value in datapoint.features.items():
            key = self._get_weight_key(feat_name, label)
            score_val += self.weights[key] * feat_value
        return score_val

    def predict(self, datapoint: DataPointWithFeatures) -> str:
        """Predicts a label for an input.

        Inputs:
            datapoint: Input data point.

        Returns:
            The predicted class.
        """
        # TODO: Implement this! Expected # of lines: <5
        best_label = None
        best_score = float("-inf")
        
        for label in self.labels:
            score = self.score(datapoint, label)
            
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def update_parameters(
        self, datapoint: DataPointWithFeatures, prediction: str, lr: float
    ) -> None:
        """Update the model weights of the model using the perceptron update rule.

        Inputs:
            datapoint: The input example, including its label.
            prediction: The predicted label.
            lr: Learning rate.
        """
        # TODO: Implement this! Expected # of lines: <10
        truth = datapoint.label
        if prediction == truth:
            return

        for feat, val in datapoint.features.items():
            true_key = self._get_weight_key(feat, truth)
            self.weights[true_key] += lr * val

            pred_key = self._get_weight_key(feat, prediction)
            self.weights[pred_key] -= lr * val

    def train(
        self,
        training_data: List[DataPointWithFeatures],
        val_data: List[DataPointWithFeatures],
        num_epochs: int,
        lr: float,
    ) -> None:
        """Perceptron model training. Updates self.weights and self.labels
        We greedily learn about new labels.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        # TODO: Implement this!
        for dp in training_data:
            self.labels.add(dp.label)

        for epoch in range(num_epochs):
            random.shuffle(training_data)
            
            for dp in training_data:
                prediction = self.predict(dp)
                self.update_parameters(dp, prediction, lr)
            
            val_acc = self.evaluate(val_data)
            print(f"Epoch {epoch+1}: Acc: {val_acc:.2%}")

    def save_weights(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps(self.weights, indent=2, sort_keys=True))
        print(f"Model weights saved to {path}")

    def evaluate(
        self,
        data: List[DataPointWithFeatures],
        save_path: str = None,
    ) -> float:
        """Evaluates the model on the given data.

        Inputs:
            data (list of Datapoint): The data to evaluate on.
            save_path: The path to save the predictions.

        Returns:
            accuracy (float): The accuracy of the model on the data.
        """
        correct = 0
        total = len(data)
        pred_label = [] 

        for dp in data:
            pred = self.predict(dp)
            pred_label.append(pred)
            if pred == dp.label:
                correct += 1

        if save_path:
            save_results(data, pred_label, save_path)
        
        if total == 0:
            total = 1
        return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="bow",
        help="Feature type, e.g., bow+len",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    args = parser.parse_args()

    data_type = DataType(args.data)
    feature_types: Set[str] = set(args.features.split("+"))
    num_epochs: int = args.epochs
    lr: float = args.learning_rate

    train_data, val_data, dev_data, test_data = load_data(data_type)
    train_data = featurize_data(train_data, feature_types)
    val_data = featurize_data(val_data, feature_types)
    dev_data = featurize_data(dev_data, feature_types)
    test_data = featurize_data(test_data, feature_types)

    model = PerceptronModel()
    print("Training the model...")
    model.train(train_data, val_data, num_epochs, lr)

    # Predict on the development set.
    dev_acc = model.evaluate(
        dev_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_{args.features}_dev_predictions.csv",
        ),
    )
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on the test set
    _ = model.evaluate(
        test_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_test_predictions.csv",
        ),
    )

    model.save_weights(
        os.path.join(
            "results", f"perceptron_{args.data}_{args.features}_model.json"
        )
    )
