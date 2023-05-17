from abc import abstractmethod
import os

import torch
import transformers
import pandas as pd


class ParaphraseDataset:
    def __init__(self, path):
        self.path = path
        self.train_sets = []
        self.val_sets = []
        self.test_sets = []
        self.train_df = None
        self.val_df = None
        self.test_df = None

    @abstractmethod
    def add_train_set(self, name):
        pass

    @abstractmethod
    def add_val_set(self, name):
        pass

    @abstractmethod
    def add_test_set(self, name):
        pass

    @abstractmethod
    def compile_dataset(self):
        pass


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

    def add_train_set(self, name):
        if name not in self.possible_sets:
            assert f"supported datasets: {self.possible_sets}"
        self.train_sets.append(pd.read_csv(os.path.join(self.path, name)))

    def add_val_set(self, name):
        if name not in self.possible_sets:
            assert f"supported datasets: {self.possible_sets}"
        self.val_sets.append(pd.read_csv(os.path.join(self.path, name)))

    def add_test_set(self, name):
        if name not in self.possible_sets:
            assert f"supported datasets: {self.possible_sets}"
        self.test_sets.append(pd.read_csv(os.path.join(self.path, name)))

    def compile_dataset(self):
        self.train_df = pd.concat(self.train_sets)
        self.val_df = pd.concat(self.val_sets)
        self.test_df = pd.concat(self.test_sets)


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
