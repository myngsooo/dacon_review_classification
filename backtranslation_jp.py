import os
import re
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pororo import Pororo
import pandas as pd

# 모델 및 데이터 로드
nmt = Pororo(task="translation", lang="multi", model="transformer.large.multi.fast.mtpg")
train = pd.read_csv('dataset/en_aug_train.csv')
test = pd.read_csv('dataset/en_aug_test.csv')

# 텍스트 전처리 (이모지 제거)
def remove_emoji(text):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 데이터셋 생성 (한국어 -> 이모지 전처리 -> 영어 -> 한국어)
def make_aug_dataset(data):
    kor_train_review_list = []
    eng_train_review_list = []

    for i, text in enumerate(data['reviews']):
        text = remove_emoji(text)

        jp_review = nmt(src="ko", text=text, tgt="ja")
        ko_review = nmt(src="ja", text=jp_review, tgt="ko")

        eng_train_review_list.append(jp_review)
        kor_train_review_list.append(ko_review)
        print(f"{i}<SEP>{jp_review}<SEP>{ko_review}")
        
    return kor_train_review_list, eng_train_review_list

# 실행
jp_ko_aug_train, jp_train = make_aug_dataset(train)
train['jp_ko_review'] = jp_ko_aug_train
train['jp_review'] = jp_train
train.to_csv('./dataset/jp_aug_train.csv', index=False)

jp_ko_aug_test, jp_test = make_aug_dataset(test)
test['jp_ko_review'] = jp_ko_aug_test
test['jp_review'] = jp_test
test.to_csv('./dataset/jp_aug_test.csv', index=False)