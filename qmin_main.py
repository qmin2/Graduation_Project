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
from transformers import \
(MT5Model, BertModel, BertTokenizer, 
 GPT2Model, GPT2Tokenizer, AutoTokenizer, set_seed, AutoModel,
 T5Tokenizer, MT5Tokenizer)
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
            # output_vectors = []
            # input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            # sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            # sum_mask = input_mask_expanded.sum(1)
            # sum_mask = torch.clamp(sum_mask, min=1e-9)

            # output_vectors.append(sum_embeddings / sum_mask)
            # output_vector = torch.cat(output_vectors, 1)
            # return output_vector
        
            return ((last_hidden * attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / attention_mask.squeeze(1).sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError
        
# according to st5, gonna followe enc-mean structure
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
        # encoder_outputs = [16, 128, 768]
        sentence_representation = self.pooler(attention_mask, encoder_outputs)

        return sentence_representation
        

# dataset의 label을 결국 integer 형태로 바꿀까?? 학습속도 빨라지게하기 위해,,, 학습속도문제가아니라 그냥 바꿔야할듯
class MBTIDataset(Dataset):
    def __init__(self, args, df, tokenizer):
        self.instances = []
        self.labels = [label for label in df['type']]

        for text in df['posts']: # 필요시 수정 이것도 gpu에서 하면되지않나?
            instance = tokenizer(text,
                                padding='max_length',
                                max_length=args.max_seq_len,
                                truncation=True,
                                return_tensors="pt")

            self.instances.append(instance)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]


class MT5Classifier(nn.Module):  # MBTI classification
    def __init__(self, args):
        super(MT5Classifier, self).__init__()
        self.args = args
        self.model = ST5_MT5Model.from_pretrained(args.model_name)

        ############ classifier만 학습시켜야 하나??, 둘다해봐야되나??
        self.model.requires_grad_(not args.freeze_lm)
        ############

        self.fcl = nn.Linear(args.hidden_size, args.num_classes) 


    def forward(self, input_ids, attention_mask):
        mt5_out = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=True)
        # mt5_out = [bsz, dim]
        linear_output = self.fcl(mt5_out)
        return linear_output


def train(args, model, tokenizer, train_data, val_data):
    train, val = MBTIDataset(args, train_data, tokenizer), MBTIDataset(args, val_data, tokenizer)

    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch_num in range(args.epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_input, train_label in tqdm(train_dataloader):  # train_input 확인
            train_label = train_label.to(args.device)
            input_ids = train_input['input_ids'].squeeze(1).to(args.device)
            attention_mask = train_input['attention_mask'].to(args.device)

            model.zero_grad()

            output = model(input_ids, attention_mask)
            # output = [bsz, num_of_classes]

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

        # after each epoch
        total_acc_val = 0
        total_loss_val = 0

        model.eval()
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(args.device)
                input_ids = val_input['input_ids'].squeeze(1).to(args.device)
                attention_mask = val_input['attention_mask'].to(args.device)
            

                output = model(input_ids, attention_mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            print(f"Epochs: {epoch_num + 1} \
            | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")

            if best_acc < (total_acc_val / len(val_data)):
                best_acc = (total_acc_val / len(val_data))
                best_model = copy.deepcopy(model)

                torch.save(
                    {
                        "model": "best_model",
                        "epoch": epoch_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "total_loss": total_loss_train,
                    },
                    f"./ckpt/best_model.pt",
                )
    
    return best_model


def evaluate(args, model, tokenizer, test_data):
    test = MBTIDataset(args, test_data, tokenizer)
    test_dataloader = DataLoader(test, batch_size=args.batch_size)

    pred_labels = []
    true_labels = []

    total_acc_test = 0

    model.eval()
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(args.device)
            input_ids = test_input['input_ids'].squeeze(1).to(args.device)
            attention_mask = test_input['attention_mask'].to(args.device)

            output = model(input_ids, attention_mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            true_labels += test_label.cpu().numpy().flatten().tolist()
            pred_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='google/mt5-base', type=str)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)

    # Hyper parameter
    parser.add_argument('--freeze_lm', default=False, action='store_true')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int) 
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-5, type=float) 
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=715, type=int)
    parser.add_argument('--patience', default = 10, type=int) # for early stop, yet 

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    set_seed(args.seed)

    model = MT5Classifier(args).to(args.device)  # 영현이형에게 물어보기, bert 처럼 특정이름만써야됐었던 논리
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base') # same as MT5Tokenizer

    df = pd.read_csv('./mbti_data/preprocessed_mbti.csv')
    #df = df[:100]

    # train 60% -> train ,,,, train 20% -> dev ,,,, train 20% -> test
    df_train, df_val = np.split(df.sample(frac=1, random_state=42), [int(0.6 * len(df))])  # random_state works as seed
    df_val, df_test = np.split(df_val.sample(frac=1, random_state=42), [int(0.5 * len(df_val))])


    best_model = train(args, model, tokenizer, df_train, df_val)
    evaluate(args, best_model, tokenizer, df_test)

    print("Finished!")

    ## checkpoint 저장해야함



if __name__ == "__main__":
    main()

