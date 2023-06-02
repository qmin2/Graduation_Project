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
from transformers import T5Model, BertModel, BertTokenizer, AutoTokenizer, GPT2Model, GPT2Tokenizer, set_seed, AutoModel
from transformers.models.t5.modeling_t5 import *
from transformers.models.gpt2.modeling_gpt2 import *
from tqdm import tqdm
from datasets import load_dataset
from mteb import MTEB
from typing import Optional, Union, List, Dict, Tuple
from tqdm import trange

### T5base Simcse train한다면 이코드로 하면 될듯?



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
        

class ST5_T5Model(T5EncoderModel):
    def __init__(self, config: T5Config, pooler_type='avg'):
        super(T5EncoderModel,self).__init__(config)
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
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
        
    
class MT5Classifier(nn.Module):  # MBTI classification
    def __init__(self,args):
        pass
    def forward(self):
        pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model_name_or_path', default='t5-base', type=str)

    ###### For MTEB ######
    # Initialize
    parser.add_argument('--task_types', nargs='*', default = None, type= str) # nargs doesn't care the order of arguments 
    parser.add_argument('--tasks', nargs='*', default = None, type= str)
    parser.add_argument('--task_categories', nargs='*', default = None, type= str)
    parser.add_argument('--task_langs', nargs='*', default=None, type=str)
    parser.add_argument('--version', default=None, type=str)
    parser.add_argument('--err_logs_path', default=None, type=str)

    # Run
    parser.add_argument('--eval_splits', nargs='*', default = None, type= str)
    parser.add_argument('--output_folder', default = None, type= str)
    ###### For MTEB ######

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.random.manual_seed(args.seed)


    # load the trained model
    backbone = ST5_T5Model.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    

    class MyModel():
        def __init__(self, backbone, tokenizer, device, **kwargs):
            self.backbone = backbone
            self.tokenizer = tokenizer
            self.device = device
            #self.model_args = kwargs["model_args"]

        def _text_length(self, text: Union[List[int], List[List[int]]]):
            """
            Help function to get the length for the input text. Text can be either
            a list of ints (which means a single text as input), or a tuple of list of ints
            (representing several text inputs to the model).
            """
            if isinstance(text, dict):              #{key: value} case
                return len(next(iter(text.values())))
            elif not hasattr(text, '__len__'):      #Object has no len() method
                return 1
            elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
                return len(text)
            else:
                return sum([len(t) for t in text])      #Sum of length of individual strings

        def encode(self, sentences, batch_size=64, **kwargs): 
            self.backbone.to(self.device)

            all_embeddings = []
            length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                features = self.tokenizer( 
                    sentences_batch,
                    return_tensors='pt',
                    max_length = 512,
                    padding=True,
                    truncation = True
                    )
                features.to(self.device)

                self.backbone.eval()
                with torch.no_grad():
                    embeddings = self.backbone(**features, output_hidden_states=True, return_dict=True)
                    embeddings = embeddings.detach().cpu()
                    #embeddings.pooler_output = [num_senteces, dim]
                    all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

            return all_embeddings
                

    model = MyModel(backbone, tokenizer, args.device)
    evaluation = MTEB(task_types=args.task_types, task_categories= args.task_categories, tasks= args.tasks, 
        task_langs=args.task_langs, version=args.version, err_logs_path=args.err_logs_path)
    results = evaluation.run(model, output_folder=args.output_folder, eval_splits=["test"])
    

if __name__=='__main__':
    main()



# according to st5, gonna followe enc-mean structure
def main():
    model = ST5_T5Model.from_pretrained('t5-base')
    

    pass


if __name__ =='main':
    main()

