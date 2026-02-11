import os
import random

from utils import DataType, load_data
from perceptron import PerceptronModel, featurize_data

def main():
    data_type = DataType("sst2")
    feature_types = {"bowc", "bigrams", "style", "lex", "neg", "con"} 
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
    dev_pred_path  = os.path.join("results", "perceptron_sst_dev.csv")
    test_pred_path = os.path.join("results", "perceptron_sst2_test_predictions.csv")
    weight_path    = os.path.join("results", "perceptron_sst_weights.json")

    dev_acc = model.evaluate(dev_data, save_path=dev_pred_path)
    print(f"SST2 perceptron accuracy: {dev_acc:.4f}")

    _ = model.evaluate(test_data, save_path=test_pred_path)
    model.save_weights(weight_path)

if __name__ == "__main__":
    main()
