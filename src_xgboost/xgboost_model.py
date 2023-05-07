import argparse
import datetime
import copy
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.mt5.modeling_mt5 import *
from transformers.models.t5.modeling_t5 import *
from transformers.models.gpt2.modeling_gpt2 import *
from tqdm import tqdm


class MBTIClassifierT5(T5Model):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()  # 사실 걍 encoder

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_lines = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        ##### evaluation할때랑 다른가?
        decoder_input_ids = kwargs["decoder_input_ids"]
        decoder_input_ids = self._shift_right(decoder_input_ids)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            # block.append(mod.module)
            block.append(mod)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        # total_length = n_lines * line_length
        bsz, total_length = input_ids.shape
        line_length = total_length // self.n_lines
        input_ids = input_ids.view(bsz * self.n_lines, line_length)
        attention_mask = attention_mask.view(bsz * self.n_lines, line_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(
            bsz, self.n_lines * line_length, -1
        )
        # outputs = BaseModelOutput(last_hidden_state=outputs[0].view(bsz, self.n_lines * line_length, -1),) + outputs[1:]
        return outputs
        # hidden_state = [1, num_dialogues * L, dim]
