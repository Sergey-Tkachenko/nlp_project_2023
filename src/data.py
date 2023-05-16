import torch
import transformers
import pandas as pd


class PairedSentenceDataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int):

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

        tokenizer_output = self.tokenizer(first_sentence, second_sentence,
                                return_tensors="pt",
                                return_token_type_ids=True,
                                max_length=self.max_length,
                                padding="max_length",
                                truncation=True)

        return {
            "labels": torch.LongTensor([label]),
            **tokenizer_output
        }


def build_tokenizer(model: str):
    return transformers.AutoTokenizer.from_pretrained(model)