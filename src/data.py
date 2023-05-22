import os
from abc import abstractmethod
from collections import namedtuple
from typing import List

import torch
import transformers
import pandas as pd


Dataset = namedtuple("Dataset", ["path", "n_rows"])


class ParaphraseDataset:
    def __init__(self, path):
        self.path = path
        self.train_sets = []
        self.val_sets = []
        self.test_sets = []
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.possible_sets = []

    def add_set(self, name: str, type: str, n: int = -1):
        if name not in self.possible_sets:
            assert f"supported datasets: {self.possible_sets}"
        if type == "train":
            self.train_sets.append(Dataset(os.path.join(self.path, name), n))
        if type == "val":
            self.val_sets.append(Dataset(os.path.join(self.path, name), n))
        if type == "test":
            self.test_sets.append(Dataset(os.path.join(self.path, name), n))

    @abstractmethod
    def concat_datasets(sets):
        pass

    def compile_dataset(self):
        self.train_df = self.concat_datasets(self.train_sets)
        self.val_df = self.concat_datasets(self.val_sets)
        self.test_df = self.concat_datasets(self.test_sets)


class PawsParaphraseDataset(ParaphraseDataset):
    def __init__(self, path: str):
        super().__init__(path)
        self.possible_sets = [
            "labeled_final_validation.csv",
            "unlabeled_final_train.csv",
            "unlabeled_final_validation.csv",
            "labeled_swap_train.csv",
            "labeled_final_train.csv",
            "labeled_final_test.csv",
        ]

    @staticmethod
    def concat_datasets(sets):
        if not sets:
            return None
        return pd.concat(
            [
                pd.read_csv(dataset_path) if n_rows < 0 else pd.read_csv(dataset_path)[:n_rows]
                for dataset_path, n_rows in sets
            ]
        )


class PawsQQPParaphraseDataset(ParaphraseDataset):
    def __init__(self, path: str):
        super().__init__(path)
        self.possible_sets = [
            "train.tsv",
            "dev_and_test.tsv",
        ]

    @staticmethod
    def concat_datasets(sets):
        if not sets:
            return None
        return pd.concat(
            [
                pd.read_csv(dataset_path, sep="\t", header=0)
                if n_rows < 0
                else pd.read_csv(dataset_path, sep="\t", header=0)[:n_rows]
                for dataset_path, n_rows in sets
            ]
        )


class DatasetManager:
    def __init__(self, datasets: List[ParaphraseDataset]):
        self.train_df = pd.concat([dataset.train_df for dataset in datasets])
        self.val_df = pd.concat([dataset.val_df for dataset in datasets])
        self.test_df = pd.concat([dataset.test_df for dataset in datasets])

        self.train_sets = []
        self.val_sets = []
        self.test_sets = []
        for dataset in datasets:
            self.train_sets.extend(dataset.train_sets)
            self.val_sets.extend(dataset.val_sets)
            self.test_sets.extend(dataset.test_sets)


class PairedSentenceDataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super().__init__()

        self.first_sentences = table["sentence1"].values
        self.second_sentences = table["sentence2"].values

        self.labels = table["label"].values

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.first_sentences)

    def __getitem__(self, index: int):
        first_sentence = self.first_sentences[index]

        second_sentence = self.second_sentences[index]

        label = self.labels[index]

        tokenizer_output = self.tokenizer(
            first_sentence,
            second_sentence,
            return_tensors="pt",
            return_token_type_ids=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        return {"labels": torch.LongTensor([label]), **tokenizer_output}


def build_tokenizer(model: str):
    return transformers.AutoTokenizer.from_pretrained(model, use_fast=True)
