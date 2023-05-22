from typing import Any
from dataclasses import dataclass

import torch
import transformers
from transformers import DebertaV2Model
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor


class BaseClassificationHead(torch.nn.Module):
    """A base class for all variants of classification heads."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any]):
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
    def __init__(self, input_hidden_size: int, size_after_pooling: int, output_perceptron_sizes: list[int]) -> None:
        super().__init__()

        self.down_size_linear_layers = torch.nn.ModuleDict()

        for name in ["mean", "max"]:
            self.down_size_linear_layers[name] = torch.nn.Sequential(
                torch.nn.Linear(input_hidden_size, size_after_pooling),
                torch.nn.BatchNorm1d(size_after_pooling),
                torch.nn.LeakyReLU(),
            )

        self.classifier = build_perceptron(output_perceptron_sizes)

    @staticmethod
    def _get_masked_mean_embeddings(embeddings: torch.Tensor, input_mask: torch.LongTensor):
        input_mask = input_mask.unsqueeze(-1)  # [batch size, seq_length, 1]

        masked_embeddings = embeddings * input_mask

        return torch.sum(masked_embeddings, dim=1) / torch.sum(input_mask, dim=1)

    @staticmethod
    def _get_masked_max_embeddings(embeddings: torch.Tensor, input_mask: torch.LongTensor):
        input_mask = input_mask.unsqueeze(-1)  # [batch size, seq_length]

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
    def __init__(
        self, conv_sizes: list[int], output_perceptron_sizes: list[int], kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()

        self.conv_net = torch.nn.Sequential()

        for prev_hid_dim, next_hid_dim in zip(conv_sizes[:-1], conv_sizes[1:]):
            self.conv_net.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=prev_hid_dim, out_channels=next_hid_dim, kernel_size=kernel_size, padding=1
                    ),
                    torch.nn.BatchNorm1d(next_hid_dim),
                    torch.nn.ReLU(),
                ]
            )

        self.classifier = build_perceptron(output_perceptron_sizes)

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        cls_embeddings = model_outputs.last_hidden_state[:, 1]

        other_embeddings = model_outputs.last_hidden_state[:, 1:]

        conv_output = torch.max(self.conv_net(other_embeddings.permute(0, 2, 1)), dim=2)[0]

        return self.classifier(torch.cat([cls_embeddings, conv_output], dim=-1))


class ConcatenatePooler(BaseClassificationHead):
    def __init__(self, output_perceptron_sizes: list[int], count_last_layers_to_use: int = 4) -> None:
        super().__init__()

        self.classifier = build_perceptron(output_perceptron_sizes)

        self.count_last_layers_to_use = count_last_layers_to_use

    @staticmethod
    def _concat_along_axis(tensor: torch.Tensor):
        """[L, BATCHS_SIZE, HID_DIM] -> [BATCH_SIZE, L * HID_DIM]"""

        return tensor.permute(1, 0, 2).reshape((tensor.shape[1], -1))

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        cls_tokens = torch.stack(model_outputs.hidden_states)[-self.count_last_layers_to_use :, :, 0]

        concatenated = ConcatenatePooler._concat_along_axis(cls_tokens)

        return self.classifier(concatenated)


@dataclass
class LSTMconfig:
    input_hid_size: int
    lstm_hid_size: int
    lstm_num_layers: int
    bidir: bool = True


class LSTMPooler(BaseClassificationHead):
    def __init__(self, output_percetpron_sizes: list[int], lstm_config: LSTMconfig) -> None:
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=lstm_config.input_hid_size,
            hidden_size=lstm_config.lstm_hid_size,
            num_layers=lstm_config.lstm_num_layers,
            batch_first=False,
            bidirectional=lstm_config.bidir,
        )

        self.classifier = build_perceptron(output_percetpron_sizes)

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        cls_hidden_states = torch.stack([hidden_state[:, 0] for hidden_state in model_outputs.hidden_states])

        _, (features_from_each_layer, _) = self.lstm(cls_hidden_states)

        concatenated_features = ConcatenatePooler._concat_along_axis(features_from_each_layer)

        return self.classifier(concatenated_features)


class TwoLSTMPooler(BaseClassificationHead):
    def __init__(
        self, layerwise_lstm_config: LSTMconfig, tokenwise_lstm_config: LSTMconfig, output_percetpron_sizes: list[int]
    ) -> None:
        super().__init__()

        self.layerwise_lstm = torch.nn.LSTM(
            input_size=layerwise_lstm_config.input_hid_size,
            hidden_size=layerwise_lstm_config.lstm_hid_size,
            num_layers=layerwise_lstm_config.lstm_num_layers,
            batch_first=False,
            bidirectional=layerwise_lstm_config.bidir,
        )

        self.tokenwise_lstm = torch.nn.LSTM(
            input_size=tokenwise_lstm_config.input_hid_size,
            hidden_size=tokenwise_lstm_config.lstm_hid_size,
            num_layers=tokenwise_lstm_config.lstm_num_layers,
            batch_first=False,
            bidirectional=tokenwise_lstm_config.bidir,
        )

        self.classifier = build_perceptron(output_percetpron_sizes)

    @staticmethod
    def _mask_sentences(
        sequence_embeddings: torch.Tensor, token_type_ids: torch.LongTensor, input_attention_mask: torch.LongTensor
    ):
        """
        Args:
            sequence_embeddings: torch.Tensor of shape [SEQ_LENGTH, BATCH_SIZE, HID_DIM],

            token_type_ids: torch.LongTensor of shape [SEQ_LENGTH, BATCH_SIZE, 1]

            attention_masks: torch.LongTensor of shape [SEQ_LENGTH, BATCH_SIZE, 1]
        """

        token_type_ids = token_type_ids.unsqueeze(-1).permute((1, 0, 2))

        input_attention_mask = input_attention_mask.unsqueeze(-1).permute((1, 0, 2))

        masked_sentences = (
            sequence_embeddings * input_attention_mask * token_type_ids,
            sequence_embeddings * input_attention_mask * (1 - token_type_ids),
        )

        return masked_sentences

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        hidden_states = torch.stack(model_outputs.hidden_states)

        layerwise_lstm_ouputs = []

        for token_index in range(hidden_states.shape[2]):
            _, (feature, _) = self.layerwise_lstm(hidden_states[:, :, token_index])

            feature = ConcatenatePooler._concat_along_axis(feature)

            layerwise_lstm_ouputs.append(feature)  # feature [BATCH_SIZE, HID_DIM]

        layerwise_lstm_ouputs = torch.stack(layerwise_lstm_ouputs)  # [SEQ_LENGTH, BATCH_SIZE, HID_DIM]

        cls_embeddings = layerwise_lstm_ouputs[0]

        masked_lstm_outputs = TwoLSTMPooler._mask_sentences(
            layerwise_lstm_ouputs[1:], inputs["token_type_ids"][:, 1:], inputs["attention_mask"][:, 1:]
        )

        sentence_embeddings = []
        for sentence_masked in masked_lstm_outputs:
            _, (layer_states, _) = self.tokenwise_lstm(sentence_masked)

            sentence_embeddings.append(ConcatenatePooler._concat_along_axis(layer_states))

        concatenated_embeddings = torch.cat([cls_embeddings] + sentence_embeddings, dim=-1)

        return self.classifier(concatenated_embeddings)


class ConcatenatePoolerWithLSTM(BaseClassificationHead):
    def __init__(
        self, tokenwise_lstm_config: LSTMconfig, output_percetpron_sizes: list[int], count_last_layers_to_use: int = 4
    ) -> None:
        super().__init__()

        self.tokenwise_lstm = torch.nn.LSTM(
            input_size=tokenwise_lstm_config.input_hid_size,
            hidden_size=tokenwise_lstm_config.lstm_hid_size,
            num_layers=tokenwise_lstm_config.lstm_num_layers,
            batch_first=False,
            bidirectional=tokenwise_lstm_config.bidir,
        )

        self.classifier = build_perceptron(output_percetpron_sizes)

        self.count_last_layers_to_use = count_last_layers_to_use

    @staticmethod
    def _mask_sentences(
        sequence_embeddings: torch.Tensor, token_type_ids: torch.LongTensor, input_attention_mask: torch.LongTensor
    ):
        """
        Args:
            sequence_embeddings: torch.Tensor of shape [SEQ_LENGTH, BATCH_SIZE, HID_DIM],

            token_type_ids: torch.LongTensor of shape [BATCH_SIZE, SEQ_LENGTH]

            attention_masks: torch.LongTensor of shape [BATCH_SIZE, SEQ_LENGTH]
        """

        token_type_ids = token_type_ids.unsqueeze(-1).permute((1, 0, 2))

        input_attention_mask = input_attention_mask.unsqueeze(-1).permute((1, 0, 2))

        masked_sentences = (
            sequence_embeddings * input_attention_mask * token_type_ids,
            sequence_embeddings * input_attention_mask * (1 - token_type_ids),
        )

        return masked_sentences

    def forward(self, model_outputs: BaseModelOutput, **inputs: dict[Any, Any]):
        hidden_states = torch.cat(model_outputs.hidden_states[-self.count_last_layers_to_use :], dim=-1).permute(
            1, 0, 2
        )  # [SEQ_LENGTH, BATCH_SIZE, 4 * HID_DIM]

        cls_embeddings = hidden_states[0]

        masked_lstm_outputs = TwoLSTMPooler._mask_sentences(
            hidden_states[1:], inputs["token_type_ids"][:, 1:], inputs["attention_mask"][:, 1:]
        )

        sentence_embeddings = []
        for sentence_masked in masked_lstm_outputs:
            _, (layer_states, _) = self.tokenwise_lstm(sentence_masked)

            sentence_embeddings.append(ConcatenatePooler._concat_along_axis(layer_states))

        concatenated_embeddings = torch.cat([cls_embeddings] + sentence_embeddings, dim=-1)

        return self.classifier(concatenated_embeddings)
