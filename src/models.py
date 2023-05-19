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
            pooler_output: [BATCH_SIZE, HIDDEN_SIZE],
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

        self.classifier = torch.nn.Sequential()

        for input_size, output_size in zip(hidden_sizes[:-2], hidden_sizes[1:-1]):
            self.classifier.append(torch.nn.Linear(input_size, output_size))

            self.classifier.append(torch.nn.BatchNorm1d(output_size))

            self.classifier.append(torch.nn.ReLU())

        self.classifier.append(torch.nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        cls_token_hidden_state = model_outputs.last_hidden_state[:, 0]

        return self.classifier(cls_token_hidden_state)