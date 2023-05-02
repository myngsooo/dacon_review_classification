# %% [markdown]
# # 쇼핑몰 리뷰 평점 분류 경진대회

# %%
import os
import re
from pororo import Pororo

import tqdm
import pandas as pd
import torch
import datasets
import glob

from glob import glob
from tqdm import tqdm

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# %% [markdown]
# # 0. Data Augmentation (BackTranslation)
# 데이터셋을 Augmentation하기 위한 방법으로, Pororo API를 활용하여 한국어->영어->한국어, 한국어->일본어->한국어 두가지 방법을 진행하였습니다.  
# 총 데이터셋이 50000개이므로 생각보다 시간이 걸립니다. 하나의 번역결과를 얻는데 평균 10초정도의 시간이 걸립니다. (GPU를 제대로 활용하지 못하는듯 합니다.)  
# 그래서 Train(영어), Test(영어), Train(일본어), Test(일본어) 4개로 나눠서 Python파일로 실행하였습니다. (25000*10초) 대략 3일정도 걸립니다..  

# %%
# 모델 및 데이터 로드
nmt = Pororo(task="translation", lang="multi")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# 텍스트 전처리 (이모지 제거)
def remove_emoji(text):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 데이터셋 생성 (한국어 -> 이모지 전처리 -> 번역어 -> 한국어)
def make_aug_dataset(data, src_lang, tgt_lang):
    kor_train_review_list = []
    eng_train_review_list = []

    for i, text in enumerate(data['reviews']):
        text = remove_emoji(text)

        en_review = nmt(src=src_lang, text=text, tgt=tgt_lang)
        ko_review = nmt(src=tgt_lang, text=en_review, tgt=src_lang)

        eng_train_review_list.append(en_review)
        kor_train_review_list.append(ko_review)
        print(f"[{i}, \"{en_review}\", \"{ko_review}\"],")
    
    return kor_train_review_list, eng_train_review_list

# %%
# Train(영어) 데이터셋 생성
ko_aug_train, en_train = make_aug_dataset(train, "ko", "en")
train['en_ko_review'] = ko_aug_train
train['en_review'] = en_train
train.to_csv('./dataset/en_aug_train.csv', index=False)

# Test(영어) 데이터셋 생성
ko_aug_test, en_test = make_aug_dataset(test, "ko", "en")
test['en_ko_review'] = ko_aug_test
test['en_review'] = en_test
test.to_csv('./dataset/en_aug_test.csv', index=False)

# %%
# Train(일본어) 데이터셋 생성
jp_ko_aug_train, jp_train = make_aug_dataset(train, "ko", "ja")
train['jp_ko_review'] = jp_ko_aug_train
train['jp_review'] = jp_train
train.to_csv('./dataset/jp_aug_train.csv', index=False)

# Test(일본어) 데이터셋 생성
jp_ko_aug_test, jp_test = make_aug_dataset(test, "ko", "ja")
test['aug_review'] = jp_ko_aug_test
test['jp_review'] = jp_test
test.to_csv('./dataset/jp_aug_test.csv', index=False)

# %% [markdown]
# 총 4개의 파이썬 파일로 분할하여 실행 후 생성된 4개의 CSV파일을 파이썬 코드로 합쳐주었습니다.  
# 합치는 코드에서 추가되는 Column을 더해서 최종 Augmentation 데이터셋을 생성했습니다. 앞으로는 학습을 진행할때 다음 데이터셋CSV를 활용합니다.

# %%
# index 추가된 경우 삭제하는 코드 #index=False를 추가안한경우는 실행안해도 무방
def remove_idx():
    en_aug = pd.read_csv('./dataset/en_aug_train.csv')
    en_aug =en_aug.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    en_aug.info()
    en_aug.to_csv('./dataset/en_aug_train.csv', index=False)

# Train데이터셋에서 en, jp 합치는 코드
def concat_train_dataset():
    en_ko = pd.read_csv('./dataset/en_aug_train.csv')
    jp_ko = pd.read_csv('./dataset/jp_aug_train.csv')
    en_ko['jp_ko_review'] = jp_ko['jp_ko_review']
    en_ko['jp_review'] = jp_ko['jp_review']
    en_ko = en_ko[['id', 'target', 'reviews', 'en_ko_review', 'jp_ko_review', 'en_review', 'jp_review']]
    en_ko.to_csv('./dataset/train_ko_en_jp.csv', index=False)

# Test데이터셋에서 en,jp 합치는 코드
def concat_test_dataset():
    en_ko = pd.read_csv('./dataset/en_aug_test.csv')
    jp_ko = pd.read_csv('./dataset/jp_aug_test.csv')
    en_ko['jp_ko_review'] = jp_ko['jp_ko_review']
    en_ko['jp_review'] = jp_ko['jp_review']
    en_ko = en_ko[['id', 'reviews', 'en_ko_review', 'jp_ko_review', 'en_review', 'jp_review']]
    en_ko.to_csv('./dataset/test_ko_en_jp.csv', index=False)

#remove_idx()
concat_train_dataset()
concat_test_dataset()

# %% [markdown]
# ## 1. 모델 설정
# 
# 
# 각각의 모델을 학습을 진행해서, 아래의 5개의 모델을 데이터셋을 [원본, Aug] 두가지로 나눠 조합하여 앙상블하였습니다.  
# "klue/bert-base",  "klue/roberta-large", "kykim/bert-kor-base", "kykim/electra-kor-base", "kykim_funnel-kor-base"  
# NUM_EPOCH에서 설정된 5에폭동안 매 에폭에 대한 Evaluation CSV를 생성합니다.

# %%
MODEL_NAME = "klue/bert-base"# "klue/roberta-large" "kykim/bert-kor-base" "kykim/electra-kor-base" "kykim_funnel-kor-base"
batch_size = 200
learning_rate = 2e-5
NUM_EPOCH = 5

# %% [markdown]
# ## 2. 데이터셋 로드

# %%
ko_en_jp = pd.read_csv('./dataset/train_ko_en_jp.csv')
ko_en_jp.info()

# %% [markdown]
# 불러온 데이터셋을 확인합니다. 2배로 Augmentation을 진행하여 75000개의 행을 가져야하지만 중복을 최소화하기 위해서, 각각의 Augmentation된 데이터에 대해 같은 행에 두고,  train데이터셋과 validation데이터셋으로 분할하였습니다. 

# %%
train_dataset, valid_dataset = train_test_split(ko_en_jp, random_state=2022, stratify=ko_en_jp['target'], test_size=0.001)
train_dataset

# %% [markdown]
# ### 2.1. 데이터셋 전처리
# 위의 데이터셋 형태에서 train_test_split을  진행한 후에 한 행에 있는 [원본,영어번역,일본어번역] 3개의 데이터를 3행으로 바꿔주었습니다.

# %%
# 각각의 데이터셋에서 review, target을 추출을 진행
train, valid = [e[['reviews', 'target']] for e in [train_dataset, valid_dataset]]
en_ko_train, en_ko_valid = [e[['en_ko_review', 'target']] for e in [train_dataset, valid_dataset]]
jp_ko_train, jp_ko_valid = [e[['jp_ko_review', 'target']] for e in [train_dataset, valid_dataset]]

en_ko_train.columns = ['reviews', 'target']
en_ko_valid.columns = ['reviews', 'target']
jp_ko_train.columns = ['reviews', 'target']
jp_ko_valid.columns = ['reviews', 'target']
en_ko_train

# %%
# 각 25000개의 행을 3개를 concat함수를 활용하여 75000개의 데이터셋으로 변형
# 원본데이터롤 진행하려면 train, valid만 활용
all_train = pd.concat([train, en_ko_train, jp_ko_train], ignore_index=True)
all_valid = pd.concat([valid, en_ko_valid, jp_ko_valid], ignore_index=True)

# %%
# 허깅페이스 데이터셋을 활용하여, 데이터셋 로드 
raw_train = Dataset.from_pandas(all_train)
raw_valid = Dataset.from_pandas(all_valid)
raw_test = load_dataset('csv', data_files='./dataset/test.csv')

# %%
# 따로 불러들인 train,valid,test데이터셋을 하나의 DatasetDict로 합치기
review_dataset = datasets.DatasetDict({'train': raw_train, 'valid': raw_valid, 'test': raw_test['train']})
review_dataset

# %% [markdown]
# ### 2.2 허깅페이스 데이터셋 로드 & 데이터로더

# %%
# 사용할 모델의 토크나이저로 테스트 및 모델의 입력가능한 형태로 변형
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(tokenizer.tokenize(en_ko_train['reviews'][0]))

def preprocess_function(example):
    return tokenizer(example["reviews"], truncation=True)
    
tokenized_datasets = review_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
tokenized_datasets

# %%
# 생성된 데이터셋에서 필요없는 Column 삭제 (id, reviews)
tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(["reviews"])
tokenized_datasets['valid'] = tokenized_datasets['valid'].remove_columns(["reviews"])
tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(["id", "reviews"])

# 정답 columns이름을 target에서 labels로 변경
tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("target", "labels")
tokenized_datasets['valid'] = tokenized_datasets['valid'].rename_column("target", "labels")

tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

# %%
from torch.utils.data import DataLoader
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
valid_dataloader = DataLoader(tokenized_datasets["valid"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, batch_size=batch_size, collate_fn=data_collator)

# %%
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# %% [markdown]
# # 3. 모델 로드

# %%
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6) # 편의상 6으로 설정/ (1,2,4,5)만 예측

# %%
from transformers import get_scheduler, AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate)

num_training_steps = NUM_EPOCH * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

# %% [markdown]
# ## 4. 모델 학습

# %%
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(NUM_EPOCH):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.save_pretrained(f"./aug_result/{MODEL_NAME}/{epoch+1}")
    tokenizer.save_pretrained(f"./aug_result/{MODEL_NAME}/{epoch+1}")

# %% [markdown]
# ## 5. 모델 검증

# %%
from torchmetrics import Accuracy

# validataion 데이터셋을 활용하여 모델 검증
def validation_model(model):
    accuracy = Accuracy()
    prediction_list_valid = []
    target_list_valid = []

    model.eval()
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu()
        targets = batch['labels'].cpu()

        prediction_list_valid.extend(predictions)
        target_list_valid.extend(targets)
    print(f'valid acc: {accuracy(torch.IntTensor(prediction_list_valid), torch.IntTensor(target_list_valid)).cpu().tolist():.4f}')

# train 데이터셋을 활용하여 validation결과와 차이 비교
def validation_train_model(model):
    accuracy = Accuracy()
    prediction_list_valid = []
    target_list_valid = []

    model.eval()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu()
        targets = batch['labels'].cpu()

        prediction_list_valid.extend(predictions)
        target_list_valid.extend(targets)
    print(f'train acc: {accuracy(torch.IntTensor(prediction_list_valid), torch.IntTensor(target_list_valid)).cpu().tolist():.4f}')

# %%
# 저장된 모델을 활용하여 Validation 데이터셋에서 성능확인
save_paths = sorted(glob(f"./aug_result/{MODEL_NAME}/*"))
for i, save_path in enumerate(save_paths):
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=6).to(device)
    print(f"{i+1}:model >> {save_path}")
    validation_model(model)

# %%
# 저장된 모델을 활용하여 Train 데이터셋에서 성능확인
save_paths = sorted(glob(f"./aug_result/{MODEL_NAME}/*"))
for i, save_path in enumerate(save_paths):
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=6).to(device)
    print(f"{i+1}:model >> {save_path}")
    validation_train_model(model)

# %% [markdown]
# Train데이터셋과 Validation데이터셋에 모델의 결과를 확인한결과 1~2에폭에서 오버피팅이 발생 (정확한 검증을 위해서는 train_test_split에서 test_size=0.1이상으로 설정 권장)

# %% [markdown]
# ## 6. 모델 Evaluation 및 Submission 파일 생성

# %%
def evaluate_submit_model(model, eval_epoch):
    prediction_list = []
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        prediction_list.extend(predictions.cpu().tolist())

    submission = pd.read_csv("dataset/sample_submission.csv")
    submission["target"] = prediction_list
    submission.to_csv(f"./submission/submission_{MODEL_NAME.replace('/', '_')}_{eval_epoch}.csv",index=False)

# %% [markdown]
# 각 에폭별 제출 코드 생성

# %%
save_paths = sorted(glob(f"./aug_result/{MODEL_NAME}/*"))
for i, save_path in enumerate(save_paths):
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=6).to(device)
    print(f"{i+1}:model >> {save_path}")
    evaluate_submit_model(model, i+1)

# %% [markdown]
# # 7. 앙상블 코드 생성
# 
# 2에서 6과정에서 생성된 다수의 제출파일을 활용하여 Hard Voting을 진행합니다.   
# 모델은 5개로 선정, 데이터셋은 [원본(25000), 전제Aug(75000)] 두가지를 선정하여, 총 5개의 csv파일을 선정하였습니다.

# %%
df_eval1 = pd.read_csv('./submission/submission_klue_bert-base_2.csv') # 원본
df_eval2 = pd.read_csv('./submission/submission_klue_roberta-large_2.csv') # 전제Aug
df_eval3 = pd.read_csv('./submission/submission_kykim_bert-kor-base_2.csv') # 전제Aug
df_eval4 = pd.read_csv('./submission/submission_kykim_electra-kor-base_2.csv') # 원본
df_eval5 = pd.read_csv('./submission/submission_kykim_funnel-kor-base_2.csv') #전제Aug

# %%
target_dict = {'review1': df_eval1['target'], 'review2': df_eval2['target'], 'review3': df_eval3['target'], 'review4': df_eval4['target'], 'review5': df_eval5['target']}
df_ensemble = pd.DataFrame(target_dict)
df_ensemble

# %%
from collections import Counter
def ensemble_data(data):
    # 1. 매열마다 모든 값을 Count, 가장 많은 빈도의 값을 추출.
    result_list = []
    for i in range(len(data)):
        frequent_value = data.iloc[i].value_counts().idxmax()
        result_list.append(frequent_value)
    return pd.DataFrame({'target':result_list})

# %%
result = ensemble_data(df_ensemble)
result

# %%
# 제출용 파일 불러오기
submission = pd.read_csv("dataset/sample_submission.csv") 
submission.head() 

# 예측 값 넣어주기
submission["target"] = result
submission.to_csv('./submission/submission_ensemble5_epoch2.csv', index=False)


