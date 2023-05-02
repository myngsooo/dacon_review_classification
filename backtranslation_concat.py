import os
import pandas as pd

os.makedirs('./dataset/aug_dataset', exist_ok=True)

# index 추가된 경우 삭제하는 코드
def remove_idx(path):
    en_aug = pd.read_csv(path)
    en_aug =en_aug.drop(columns=['Unnamed: 0'])#, 'Unnamed: 0.1'])
    en_aug.info()
    en_aug.to_csv(path, index=False)

# Train데이터셋에서 en, jp 합치는 코드
def concat_train_dataset():
    en_ko_jp = pd.read_csv('./dataset/jp_aug_train.csv')
    en_ko = pd.read_csv('./dataset/en_aug_train.csv')
    en_ko_jp['en_ko_review'] = en_ko['en_ko_review']
    en_ko_jp['en_review'] = en_ko['en_review']
    en_ko_jp = en_ko_jp[['id', 'target', 'reviews', 'en_ko_review', 'jp_ko_review', 'en_review', 'jp_review']]
    en_ko_jp.to_csv('./dataset/aug_dataset/aug_train_ko_en_jp.csv', index=False)

# Test데이터셋에서 en,jp 합치는 코드
def concat_test_dataset():
    en_ko_jp = pd.read_csv('./dataset/jp_aug_test.csv')
    en_ko = pd.read_csv('./dataset/en_aug_test.csv')
    en_ko_jp['en_ko_review'] = en_ko['en_ko_review']
    en_ko_jp['en_review'] = en_ko['en_review']
    en_ko_jp = en_ko_jp[['id', 'reviews', 'en_ko_review', 'jp_ko_review', 'en_review', 'jp_review']]
    en_ko_jp.to_csv('./dataset/aug_dataset/aug_test_ko_en_jp.csv', index=False)

# 잘못 생성된 index 삭제용도
# remove_idx('./dataset/jp_aug_train.csv')
# remove_idx('./dataset/jp_aug_test.csv')
# remove_idx('./dataset/en_aug_train.csv')
# remove_idx('./dataset/en_aug_test.csv')

concat_train_dataset()
concat_test_dataset()