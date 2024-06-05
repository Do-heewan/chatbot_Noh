import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import threading
import json
import pandas as pd
import tensorflow as tf
import torch

from Source.BotServer import BotServer
from Source.preprocess import Preprocess
from Source.FindAnswer import FindAnswer
from model.IntentModel import IntentModel
from qna.create_embedding_data import create_embedding_data

# 전처리 객체 생성
try:
    p = Preprocess(word2index_dic='./dict/chatbot_dict.bin',
                   userdic='./Source/user_dic.tsv')
    print("텍스트 전처리기 로드 완료..")
except: print("텍스트 전처리기 로드 실패..")

# 의도 파악 모델
try:
    intent = IntentModel(model_name='./model/intent_model.h5', preprocess=p)
    print("의도 파악 모델 로드 완료..")
except: print("의도 파악 모델 로드 실패..")

#엑셀 파일 로드
try:
    df = pd.read_excel('./qna/train_data.xlsx', engine='openpyxl')
    print("엑셀 파일 로드 완료..")
except: print("엑셀 파일 로드 실패..")

# pt 파일 갱신 및 불러오기
try:
    create_embedding_data = create_embedding_data(df=df, preprocess=p)
    create_embedding_data.create_pt_file()
    embedding_data = torch.load('./qna/embedding_data.pt')
    print("임베딩 pt 파일 갱신 및 로드 완료..")
except: print("임베딩 pt 파일 갱신 및 로드 실패..")


def to_client(conn, addr):
    try:
        # 데이터 수신
        read = conn.recv(2048) # 수신 데이터가 있을 때까지 블로킹
        print('======================')
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)  # 스레드 강제 종료

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 파악
        intent_pred = intent.predict_class(query)
        intent_name = intent.labels[intent_pred]

        # 답변 검색
        f = FindAnswer(df=df, embedding_data=embedding_data ,preprocess=p)
        selected_qes, score, answer, imageUrl, query_intent = f.search(query, intent_name)

        send_json_data_str = {
            "Query": selected_qes,
            "Answer": answer,
            "imageUrl": imageUrl,
            "Intent": intent_name
        }
        message = json.dumps(send_json_data_str) # json객체 문자열로 반환
        conn.send(message.encode()) # 응답 전송
        print(message)

    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    # 봇 서버 동작
    port = 5050
    listen = 1000
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start..")

    while True:
        conn, addr = bot.ready_for_client()
        client = threading.Thread(target=to_client, args=(
            conn,   # 클라이언트 연결 소켓
            addr,   # 클라이언트 연결 주소 정보
        ))
        client.start()