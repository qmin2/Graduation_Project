import numpy as np
import pandas as pd

import os.path

def main():
    path = './mbti_data/preprocessed_mbti.csv'
    if os.path.isfile(path):
        print("data file already exits")
        return
    
    df = pd.read_csv('./mbti_data/mbti.csv')
    df = df.replace({'type':'INFP'}, 0)
    df = df.replace({'type':'INFJ'}, 1)
    df = df.replace({'type':'INTP'}, 2)
    df = df.replace({'type':'INTJ'}, 3)
    df = df.replace({'type':'ISFP'}, 4)
    df = df.replace({'type':'ISFJ'}, 5)
    df = df.replace({'type':'ISTP'}, 6)
    df = df.replace({'type':'ISTJ'}, 7)
    df = df.replace({'type':'ENFP'}, 8)
    df = df.replace({'type':'ENFJ'}, 9)
    df = df.replace({'type':'ENTP'}, 10)
    df = df.replace({'type':'ENTJ'}, 11)
    df = df.replace({'type':'ESFP'}, 12)
    df = df.replace({'type':'ESFJ'}, 13)
    df = df.replace({'type':'ESTP'}, 14)
    df = df.replace({'type':'ESTJ'}, 15)

    df.to_csv("./mbti_data/preprocessed_mbti.csv", index= False)


if __name__ == "__main__":
    main()