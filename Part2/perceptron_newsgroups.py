import os
import random

from utils import DataType, load_data
from perceptron import PerceptronModel, featurize_data

def main():
    random.seed(0)

    data_type = DataType("newsgroups")
    feature_types = {"bowc", "bigrams","domain"} 
    num_epochs = 3 
    lr = 0.1

    train_data, val_data, dev_data, test_data = load_data(data_type)
    train_data = featurize_data(train_data, feature_types)
    val_data   = featurize_data(val_data, feature_types)
    dev_data   = featurize_data(dev_data, feature_types)
    test_data  = featurize_data(test_data, feature_types)

    model = PerceptronModel()
    model.train(train_data, val_data, num_epochs=num_epochs, lr=lr)
    os.makedirs("results", exist_ok=True)
    dev_pred_path  = os.path.join("results", "perceptron_newsgroups_dev.csv")
    test_pred_path = os.path.join("results", "perceptron_newsgroups_test_predictions.csv")
    weight_path    = os.path.join("results", "perceptron_newsgroups_weights.json")

    dev_acc = model.evaluate(dev_data, save_path=dev_pred_path)
    print(f"newsgroups perceptron accuracy: {dev_acc:.4f}")

    _ = model.evaluate(test_data, save_path=test_pred_path)
    model.save_weights(weight_path)

if __name__ == "__main__":
    main()
