# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from collections import namedtuple


class MBTIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        dialog_prefix="mbti classification:",
    ):
        self.data = data
        self.dialog_prefix = dialog_prefix
        self.Example = namedtuple("Example", ["id", "dialog", "target"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example_data = self.data[index]
        id = example_data["id"]
        dialog = [self.dialog_prefix + text for text in example_data["dialog_text"]]
        target = example_data["mbti"]

        # namedTuple안쓰면 받지를 못함;;
        return self.Example(id=id, dialog=dialog, target=target)

    def get_example(self, index):  # for checking
        return self.data[index]


class MBTICollator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, features):
        dialogs = []
        labels = []
        for example in features:
            dialogs.extend(example[1])
            labels.append(example[2])

        dialog = self.tokenizer(
            dialogs,
            return_tensors="pt",
            padding=True,  # Is it right?
            truncation=True,
            max_length=self.args.max_length,
        )
        input_ids = dialog["input_ids"]
        attention_mask = dialog["attention_mask"]
        # print(features[0][0])
        # print(input_ids.shape)

        label = self.tokenizer(  # 어차피 length 1 짜리
            labels,
            return_tensors="pt",
        )
        label_ids = label["input_ids"]
        # scalar = torch.tensor([0])
        # tensor2 = torch.cat((scalar, label_ids), dim=0)
        label_mask = label["attention_mask"]

        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            # "labels": label_ids,  # need to check # 나중에 label로 주는게 아니라, 이에대한 설명을 늘여뜨려서 embedding으로?
            "decoder_input_ids": label_ids,  # 그냥 start token 줘야되나?
        }
