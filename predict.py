#coding:utf-8
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

from load_data import preprocess, load_datasets

import pickle
from scipy import sparse
#X_train_data, y_train, X_test_data, y_test = load_datasets()
#print("参数加载完成")
vectorizer = pickle.load(open("./para/vectorizer.pickle", "rb"))
X_train_tfidf = sparse.load_npz('./para/X_train_tfidf.npz')

a=np.load('.\para\y_train.npy')
y_train=a.tolist()

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

news_lastest = ["360金融旗下产品有360借条、360小微贷、360分期。360借条是360金融的核心产品，是一款无抵押、纯线上消费信贷产品，为用户提供即时到账贷款服务（通俗可以理解为“现金贷”）用户借款主要用于消费支出。从收入构成来看，360金融主要有贷款便利服务费、贷后管理服务费、融资收入、其他服务收入等构成。财报披露，营收增长主要是由于贷款便利化服务费、贷款发放后服务费和其他与贷款发放量增加相关的服务费增加。",
                "检方并未起诉全部涉嫌贿赂的家长，但起诉名单已有超过50人，耶鲁大学、斯坦福大学等录取率极低的名校涉案也让该事件受到了几乎全球的关注，该案甚至被称作美国“史上最大招生舞弊案”。",
                "俄媒称，目前尚不清楚特朗普这一言论的指向性，因为近几日，伊朗官员们都在表达力图避免与美国发生军事冲突的意愿。5月19日早些时候，伊朗革命卫队司令侯赛因·萨拉米称，伊朗只想追求和平，但并不害怕与美国发生战争。萨拉米称，“我们（伊朗）和他们（美国）之间的区别在于，美国害怕发生战争，缺乏开战的意志。”"
                ,"瑞典斯德哥尔摩当地时间2021年10月7日13：00（北京时间19：00），瑞典学院将2021年度诺贝尔文学奖颁给了坦桑尼亚作家阿卜杜勒拉扎克·古尔纳（Abdulrazak Gurnah）。授奖词为：“鉴于他对殖民主义的影响以及文化与大陆之间的鸿沟中难民的命运的毫不妥协和富有同情心的洞察。古尔纳（1948年出生在桑给巴尔），今年73岁，坦桑尼亚小说家，以英语写作，现居英国。他最著名的小说是《天堂》（1994），它同时入围了布克奖和惠特布莱德奖，《遗弃》（2005）和《海边》（2001）则入围了布克奖和洛杉矶时报图书奖的候选名单。"
                ,"　原标题：习近平对全军后勤工作会议作出重要指示强调 加快推动现代后勤高质量发展 为实现建军一百年奋斗目标提供有力支撑新华社11月23日电（记者梅常伟）全军后勤工作会议11月22日至23日在京召开。中共中央总书记、国家主席、中央军委主席习近平作出重要指示，向全军后勤战线全体同志致以诚挚的问候。习近平强调，党的十八大以来，全军后勤战线坚决贯彻党中央和中央军委决策部署，聚焦保障打赢，积极改革创新，着力建设一切为了打仗的后勤，为我军建设发展和有效履行使命任务作出了重要贡献。希望同志们深入贯彻新时代党的强军思想，深入贯彻新时代军事战略方针，加快推进“十四五”规划任务落实，加快建设现代军事物流体系和军队现代资产管理体系，加快推动现代后勤高质量发展，为实现建军一百年奋斗目标提供有力支撑。"]

X_new_data = [preprocess(doc) for doc in news_lastest]
X_new_tfidf = vectorizer.transform(X_new_data)


predicted = classifier.predict(X_new_tfidf)
print(predicted)