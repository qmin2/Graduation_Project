import argparse
import datetime
import copy
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import MT5Model, BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, set_seed, AutoModel
from transformers.models.mt5.modeling_mt5 import *
from transformers.models.gpt2.modeling_gpt2 import *
from tqdm import tqdm
from datasets import load_dataset



class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'first': the fisrt representation of the last hidden state
    'avg': average of the last layers' hidden states at each token.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["first", "avg"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        
        if self.pooler_type == 'fisrt': # since T5 has no CLS token
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError
        

class ST5_MT5Model(MT5EncoderModel):
    def __init__(self, config: MT5Config, pooler_type='avg'):
        super(MT5EncoderModel,self).__init__(config)
        # 이렇게하면 MT5Model의 부모클래스 MT5PreTrainedModel을 상속하는거여서
        # 원래 MT5 init에 있던 코드를 다 복붙 하는것인듯.
        # 그렇다면 왜 이런식으로해서 MT5PretrainedModel의 변수까지 가져오는걸까?
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)
        self.pooler = Pooler(pooler_type)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
        >>> model = MT5EncoderModel.from_pretrained("mt5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sentence_representation = self.pooler(attention_mask, encoder_outputs)

        return sentence_representation
        

# dataset의 label을 결국 integer 형태로 바꿀까?? 학습속도 빨라지게하기 위해
class MBTIDataset(Dataset):
    pass


class MT5Classifier(nn.Module):  # MBTI classification
    def __init__(self, args):
        super(MT5Classifier, self).__init__()
        self.args = args
        self.model = ST5_MT5Model.from_pretrained(args.model_name)
        for block in self.model.h:
            block.attn.set_causal_mask(self.args.deactivate_causal_mask)
        self.model.requires_grad_(not args.freeze_lm)
        self.model.wpe.requires_grad_(False)
        if args.half_dropout_rate:
            self.model.drop.p /= 2
        self.fcl = nn.Linear(args.hidden_size, args.num_classes)


    def forward(self, input_ids, attention_mask, position_ids):
        gpt_out, _ = self.model(self.args,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                return_dict=False)  # ([16, 128, 768])
        batch_size = gpt_out.shape[0]
        if self.args.pooling_method == 'last':
            last_indices = attention_mask.squeeze().sum(dim=-1) - 1
            sent_repr = gpt_out[[i for i in range(batch_size)], last_indices]
        elif self.args.pooling_method == 'avg':
            orig_length = torch.div(attention_mask.squeeze().sum(dim=-1), 2, rounding_mode='floor')
            sent_repr = []
            for b in range(batch_size):
                sent_repr.append(gpt_out[b][orig_length[b]:orig_length[b] * 2].mean(dim=0))
            sent_repr = torch.stack(sent_repr, dim=0)
        else:
            raise Exception('Unimplemented pooling method')
        linear_output = self.fcl(sent_repr)
        return linear_output



def train():
    pass

def evalutate():
    pass

# according to st5, gonna followe enc-mean structure
def main():
    model = ST5_MT5Model.from_pretrained('mt5-base')

    df = pd.read_csv('./mbit_data/mbit.csv') # delimiter='\t' 꼭써야되나?

    # train 60% -> train ,,,, train 20% -> dev ,,,, train 20% -> test
    df_train, df_val = np.split(df.sample(frac=1, random_state=42), [int(0.6 * len(df))])
    df_val, df_test = np.split(df_val.sample(frac=1, random_state=42), [int(0.5 * len(df_val))])

    
    

    pass


if __name__ =='main':
    main()

