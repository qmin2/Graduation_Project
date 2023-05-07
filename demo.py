import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from transformers import (
    T5Tokenizer,
    T5Model,
)
from transformers.models.t5.modeling_t5 import *
from tqdm import tqdm
from src_xgboost.xgboost_model import *

import pickle
import xgboost as xgb


def main():
    input_answers = [
        "I don't want to study",
        "I always cry when I am sad",
    ]  # will have 10 answers

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    feature_data = tokenizer(
        input_answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    t5 = T5Model.from_pretrained("t5-base")
    model = MBTIClassifierT5(t5.config)
    model.load_t5(t5.state_dict())

    # dataloader = DataLoader(feature_data)

    features = []

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=feature_data["input_ids"].unsqueeze(0),
            attention_mask=feature_data["attention_mask"].unsqueeze(0),
            decoder_input_ids=torch.tensor([[0]]),
        )
        outputs = outputs.last_hidden_state[:, 0, :].numpy()
        features.extend(outputs)
        # 앞에[0] 짤라야 할듯? 뒤에 1 eos 계속 따라다님

    test_features = np.vstack(features)  # (10,768) features embedding 저장해두자

    ######## prediciton ########
    # load the saved model from a file
    ei_model = pickle.load(open("selected_tree/ei_classifier.sav", "rb"))
    ns_model = pickle.load(open("selected_tree/ns_classifier.sav", "rb"))
    tf_model = pickle.load(open("selected_tree/tf_classifier.sav", "rb"))
    pj_model = pickle.load(open("selected_tree/pj_classifier.sav", "rb"))

    # make predictions using the loaded model
    ei_pred = ei_model.predict(test_features)  # e=0, i=1
    ns_pred = ns_model.predict(test_features)  # n=0, s=1
    tf_pred = tf_model.predict(test_features)  # t=0, f=1
    pj_pred = pj_model.predict(test_features)  # p=0, j=1

    pred = str(ei_pred[0]) + str(ns_pred[0]) + str(tf_pred[0]) + str(pj_pred[0])
    mbti_dict = {
        "0000": "ENTP",
        "0001": "ENTJ",
        "0010": "ENFP",
        "0011": "ENFJ",
        "0100": "ESTP",
        "0101": "ESTJ",
        "0110": "ESFP",
        "0111": "ESFJ",
        "1000": "INTP",
        "1001": "INTJ",
        "1010": "INFP",
        "1011": "INFJ",
        "1100": "ISTP",
        "1101": "ISTJ",
        "1110": "ISFP",
        "1111": "ISFJ",
    }

    print(mbti_dict[pred])


if __name__ == "__main__":
    main()
