# multilayer_perceptron_sst.py
import os
import random
import torch

from pprint import pprint

from utils import DataType, load_data, save_results
from multilayer_perceptron import (
    Tokenizer,
    BOWDataset,
    get_label_mappings,
    MultilayerPerceptronModel,
    Trainer,
)

def main():

    data_type = DataType("newsgroups")
    max_vocab_size = 20000
    max_length = 100

    num_epochs = 10
    lr = 1e-3

    train_data, val_data, dev_data, test_data = load_data(data_type)
    tokenizer = Tokenizer(train_data, max_vocab_size=max_vocab_size)
    label2id, id2label = get_label_mappings(train_data)

    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds   = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds   = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds  = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
    )

    trainer = Trainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer.train(train_ds, val_ds, optimizer, num_epochs)

    dev_acc = trainer.evaluate(dev_ds)
    print(f"MLP newsgroup accuracy: {100 * dev_acc:.2f}%")
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]

    save_results(
        test_data,
        test_preds,
        os.path.join("results", "mlp_newsgroups_test_predictions.csv"),
    )

if __name__ == "__main__":
    main()
