# Generate A.I Model
# >> Model: 긍부정 분ㄴ석 모델(감정분석)
# >> Module: Tensorflow, Keras
# >> Dataset: Naver Sentiment Movie Corpus




# 데이터셋: Nave Sentiment Movie Corpus(https://github.com/e9t/nsmc/)
# >> 네이버 영화 리뷰 중 영화단 100개의 리뷰를 모아
# >> 총 200,000개의 리뷰(훈련: 15만개, 테스트: 5만개)로
# >> 이루어져있고, 1~10점까지의 평점 중 중립적인 평점(5~8)은
# >> 제외하고 1~4점은 부정, 9~10점을 긍정으로 동일한 비율로
# >> 데이터에 포함시킴

# >> 데이터는 id, document, label 세개의 열로 이루어져있음
# >> id: 리뷰의 고유한 Key값
# >> document: 리뷰의 내용
# >> label: 긍정(1)인지 부정(0)인지 나타냄
#           평점이 긍정(9~10졈), 부정(1~4점), 5~8점은 제거

import json
import os
import nltk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from pprint import pprint
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

#############
# File Open #
#############

# .txt 파일에서 데이터를 불러오는 method
def read_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]


        # data = []
        # for line in f.read().splitlines():
            # line = 9976970 \t	아 더빙.. 진짜 짜증나네요 목소리 \t 0
            # line,split('\t') [9976970, 아 더빙.. 진짜 짜증나네요 목소리, 0]
            # data.append(line,split('\t'))



        data = data[1:]  # 제목열 제외
    return data

# nsmc 데이터를 불러와서 python 변수에 담기
train_data = read_data('./dataset/ratings_train.txt')  # 트레이닝 데이터 Open
test_data = read_data('./dataset/ratings_test.txt')

'C:/cnu_workspace/MovieTomatoes/as/dataset/ratings_train.txt'  # 절대경로

# / - 하위폴더
# .. - 상위폴더
# . - 현재폴더

# ./datset/ratings_train.txt


# 절대경로와 상대경로
# C:/cnu_workspace/MovieTomatoes
#                   ㄴ ai
#                       ㄴ dataset
#                           ㄴ ratings_text.txt
#                           ㄴ ratings_train.txt
#                   ㄴ ModelLearning.py
#                   ㄴ model
#                   ㄴ webcrawl
#                   ㄴ main.py
#                   ㄴ README.md



# print(len(train_data))
# print(train_data[0])
#
# print(len(test_data))
# print(test_data[0])

#################
# PreProcessing #
#################
# 데이터를 학습하기에 알맞게 처리해보자. konlpy 라이브러리를 사용해서
# 형태소 분석 및 품사 태깅을 진행한다. 네이버 영화 데이터는
# 맞춤법이나 띄어쓰기가 제대로 되어있지 않은 경우가 있기 때문에
# 정확한 분류를 위해서 konlpy를 사용한다.
# konlpy는 여러 클래스가 존재하지만 그중 okt(open korean text)를
# 사용하여 간단한 문장분석을 실행한다.
okt = Okt()
# print(okt.pos('이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

# Train, Test 데이터셋에 형태소 분석과 품사 태깅 작업 진행
# norm: 그래욬ㅋㅋ - 그래요
# stem: 원형을 찾음 (그래요 - 그렇다)
def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    # 전처리 작업이 완료 된 train_docs.json 파일이 있을 때
    # train_docs.json과 test_docs.json 파일 로드!, json = dict
    with open('train_docs.json', 'r', encoding='UTF-8') as f:  # UTF-8 = 한글이므로
        train_docs = json.load(f)
    with open('test_docs.json', 'r', encoding='UTF-8') as f:
        test_docs = json.load(f)
else:
    # 전처리 된 파일이 없을 때
    # 전처리 작업 시작!
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # 전처리 완료 - JSON 파일로 저장
    with open('train_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent='\t')
    with open('test_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent='\t')

# 전처리 작업 데이터 확인
pprint(train_docs[0])
pprint(test_docs[0])
print(len(train_docs))
print(len(test_docs))

# 분석한 데이터의 토큰(문자열 분석을 위한 작은 단위)의 개수를 확인
token = [t for d in train_docs for t in d[0]]
print(len(tokens))

# 이 데이터를 nltk 라이브러를 통해서 전처리,
# vocab().most_common을 이용해서 가장 자주 사용되는 단어 빈도수 확인
text = nltk.Text(tokens, name= 'NSMC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))