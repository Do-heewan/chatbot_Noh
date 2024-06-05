from Source.preprocess_1 import Preprocess

sent = "컴공 과사 번호 알려줘...!@#%%^&*() 노희완"

# 전처리 객체 생성
p = Preprocess(userdic='../Source/user_dic.tsv')

# 형태소 분석기 실행
pos = p.pos(sent)

# 품사 태크과 같이 단어 출력
ret = p.get_keywords(pos, without_tag = False)
print(ret)

# 품사 태크 없이 단어 출력
ret = p.get_keywords(pos, without_tag = True)
print(ret)