from feature_mining_rake import *
from process import *
from tqdm import tqdm
from multiprocessing import Pool 

#对于所有的商家提取出review，然后运用Rake算法挖掘feature，以及情感分析
p = Process()
business_list = p.business_review_count_sort(dic=p.business_review_count())
# print(business_list)
# nlp = spacy.load('en')
print('Start')

def pro(muti_para):
    REVIEW_PATH = muti_para[0]
    REVIEW_PATH_WITH_ID = muti_para[1]
    try:
        with codecs.open(REVIEW_PATH, encoding='utf-8') as f:
            text = f.read()
            clean_text = text_clean(text)
            # print(clean_text)
            '''
            几个参数说明：
            stop_words_path:停用词路径
            min_char_length:feature字符至少的长度
            max_words_length:feature最多包含的单词数
            min_keyword_frequency:关键词（feature的成员）出现的频率,不能每个单词都小于这个数
            min_words_length_adj:邻接feature最少单词数
            max_words_length_adj:邻接feature最多单词数
            min_phrase_freq_adj：邻接feature在文本中至少出现的次数
            '''

            rake = Rake(stop_words_path=STOP_WORDS_PATH,
                        min_char_length=2, 
                        max_words_length=3, 
                        min_keyword_frequency=2, 
                        min_words_length_adj=2, 
                        max_words_length_adj=3, 
                        min_phrase_freq_adj=2)
            keywords = rake.run(clean_text)
            message = feature_pruning_and_sentiment_analysis(keywords, REVIEW_PATH_WITH_ID)

            write_feature_into_file(message)
    except Exception as e:
        print('没有该文件 {}搭配{}'.format(REVIEW_PATH, REVIEW_PATH_WITH_ID))

muti_para = []
for item in business_list:
    business_id_string = item[0]
    REVIEW_PATH = 'date/all_business_reviews/' + business_id_string + '.txt'
    REVIEW_PATH_WITH_ID = 'date/all_business_reviews/' + business_id_string + '_with_id_.txt'
    content = []
    content.append(REVIEW_PATH)
    content.append(REVIEW_PATH_WITH_ID)
    muti_para.append(content)

pool = Pool()
cnt = 1
for i in pool.imap(pro, muti_para):
    cnt += 1
    print('正在处理第 {} 个business'.format(cnt))
pool.close()
pool.join()


# for i in tqdm(range(len(business_list))):
#     business_id_string = business_list[i][0]
#     REVIEW_PATH = 'date/all_business_reviews/' + business_id_string + '.txt'
#     REVIEW_PATH_WITH_ID = 'date/all_business_reviews/' + business_id_string + '_with_id_.txt'
#     try:
#         with codecs.open(REVIEW_PATH, encoding='utf-8') as f:
#             text = f.read()
#             clean_text = text_clean(text)
#             # print(clean_text)
#             '''
#             几个参数说明：
#             stop_words_path:停用词路径
#             min_char_length:feature字符至少的长度
#             max_words_length:feature最多包含的单词数
#             min_keyword_frequency:关键词（feature的成员）出现的频率,不能每个单词都小于这个数
#             min_words_length_adj:邻接feature最少单词数
#             max_words_length_adj:邻接feature最多单词数
#             min_phrase_freq_adj：邻接feature在文本中至少出现的次数
#             '''

#             rake = Rake(stop_words_path=STOP_WORDS_PATH,
#                         min_char_length=2, 
#                         max_words_length=3, 
#                         min_keyword_frequency=2, 
#                         min_words_length_adj=2, 
#                         max_words_length_adj=3, 
#                         min_phrase_freq_adj=2)
#             keywords = rake.run(clean_text)
#             message = feature_pruning_and_sentiment_analysis(keywords)

#             write_feature_into_file(message)
#     except Exception as e:
#         print('没有该文件')


