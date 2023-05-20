from typing import Any

import torch
import transformers
from transformers import DebertaV2Model
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor


class BaseClassificationHead(torch.nn.Module):
    """A base class for all variants of classification heads."""

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, model_outputs: BaseModelOutput,
                **inputs: dict[Any]):
        """
        Inputs have following shapes:
            last_hidden_state: [BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
            hidden_states: [NUMBER_LAYERS, BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        
        Output tensor must have following shape: [BATCH_SIZE, 2]
        """

        raise NotImplementedError


class DebertaV2WithCustomClassifier(torch.nn.Module):
    def __init__(self, deberta_model: DebertaV2Model, classifier_head: BaseClassificationHead) -> None:
        super().__init__()

        self.deberta = deberta_model

        self.classifier_head = classifier_head
    
    def forward(self, **deberta_inputs: dict):
        deberta_outputs = self.deberta(**deberta_inputs, output_hidden_states=True)

        return self.classifier_head(deberta_outputs, **deberta_inputs)

    @property
    def device(self):
        return self.deberta.device
    

def build_perceptron(hidden_sizes: list[int]):
    
    classifier = torch.nn.Sequential()

    for input_size, output_size in zip(hidden_sizes[:-2], hidden_sizes[1:-1]):
        classifier.append(torch.nn.Linear(input_size, output_size))

        classifier.append(torch.nn.BatchNorm1d(output_size))

        classifier.append(torch.nn.LeakyReLU())

    classifier.append(torch.nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))

    return classifier
    

class PerceptronPoooler(BaseClassificationHead):
    """
    Simple classifiecation head that uses only last hidden state of CLS token.

    Almost the same compared to DebertaV2ForSequenceClassification, but allows
    several linear layers in classifier.
    """


    def __init__(self, hidden_sizes: list[int]) -> None:
        """
        Args:
            hidden_sizes: list[int] -- a list with configuration parameters for classification head. E.g. [100, 10, 2]
            corresponds to two linear layers: 100 -> 10, 10 -> 2.
        
        """
        super().__init__()

        self.classifier = build_perceptron(hidden_sizes)

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        cls_token_hidden_state = model_outputs.last_hidden_state[:, 0]

        return self.classifier(cls_token_hidden_state)


class MeanMaxPooler(BaseClassificationHead):
    def __init__(self, input_hidden_size : int, size_after_pooling: int,
                 output_perceptron_sizes: list[int]) -> None:
        super().__init__()

        
        
        self.down_size_linear_layers = torch.nn.ModuleDict()

        for name in ["mean", "max"]:
            self.down_size_linear_layers[name] = torch.nn.Sequential(
                torch.nn.Linear(input_hidden_size, size_after_pooling),
                torch.nn.BatchNorm1d(size_after_pooling),
                torch.nn.LeakyReLU()
            )

        self.classifier = build_perceptron(output_perceptron_sizes)

    @staticmethod
    def _get_masked_mean_embeddings(embeddings: torch.Tensor, input_mask: torch.LongTensor):
        input_mask = input_mask.unsqueeze(-1) # [batch size, seq_length, 1]

        masked_embeddings = embeddings * input_mask

        return torch.sum(masked_embeddings, dim=1) / torch.sum(input_mask, dim=1)
    
    @staticmethod
    def _get_masked_max_embeddings(embeddings: torch.Tensor, input_mask: torch.LongTensor):
        input_mask = input_mask.unsqueeze(-1) # [batch size, seq_length]

        masked_embeddings = torch.where(input_mask.bool(), embeddings, -1e9)

        return torch.max(masked_embeddings, dim=1)[0]

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[str, torch.Tensor]):
        # process CLS embeddings separately
        cls_embeddings = model_outputs.last_hidden_state[:, 1]

        other_embeddings = model_outputs.last_hidden_state[:, 1:]
        input_mask = inputs["attention_mask"][:, 1:]

        mean_embeddings = self._get_masked_mean_embeddings(other_embeddings, input_mask)
        max_embeddings = self._get_masked_max_embeddings(other_embeddings, input_mask)

        # reduce the size of pooled embeddings

        mean_embeddings = self.down_size_linear_layers["mean"](mean_embeddings)
        max_embeddings = self.down_size_linear_layers["max"](max_embeddings)

        # concat all result into one tensor

        all_features = torch.cat([cls_embeddings, mean_embeddings, max_embeddings], dim=1)

        return self.classifier(all_features)
    

class ConvPooler(BaseClassificationHead):
    def __init__(self, conv_sizes: list[int], output_perceptron_sizes: list[int],
                 kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()

        self.conv_net = torch.nn.Sequential()

        for prev_hid_dim, next_hid_dim in zip(conv_sizes[:-1], conv_sizes[1:]):
            self.conv_net.extend(
                [torch.nn.Conv1d(in_channels=prev_hid_dim, out_channels=next_hid_dim,
                                kernel_size=kernel_size, padding=1),
                torch.nn.BatchNorm1d(next_hid_dim),
                torch.nn.ReLU()]
            )
        
        self.classifier = build_perceptron(output_perceptron_sizes)
    
    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        cls_embeddings = model_outputs.last_hidden_state[:, 1]

        other_embeddings = model_outputs.last_hidden_state[:, 1:]

        conv_output = torch.max(self.conv_net(other_embeddings.permute(0, 2, 1)), dim=2)[0]

        return self.classifier(torch.cat([cls_embeddings, conv_output], dim=-1))
