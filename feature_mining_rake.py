import codecs
import re
import operator
import six
from six.moves import range
from collections import Counter
import io
import nltk
# from nltk.stem.porter import PorterStemmer  
# from nltk.corpus import wordnet as wn
import spacy
from textblob import TextBlob
import json
import string
import time
from multiprocessing import Pool 

# REVIEW_PATH = 'date/this_turn_review.txt'
# REVIEW_PATH_WITH_ID = 'date/this_turn_review_with_id.txt'
STOP_WORDS_PATH = 'date/SiriusStoplist.txt'
BUSINESS_FILE_PATH = 'date/yelp_academic_dataset_business.json'
USER_FILE_PATH = 'date/yelp_academic_dataset_user.json'
FEATURE_MESSAGE = 'date/feature_message.json'

debug = False
test = False
nlp = spacy.load('en')

list_temp = []

def text_clean(text):
    text = text.replace('\\n', '')
    text = text.replace('\\r', '')
    text = text.replace("'",'')
    sentence_clean = []
    for sentence in split_sentences(text):
        sentences_nlp = nlp(sentence)
        word_list = [token.lemma_ for token in sentences_nlp]
        
        sentence = ' '.join(word_list)
        sentence_clean.append(sentence)
    sentence_clean_str = '.'.join(sentence_clean)
    sentence_clean_str = sentence_clean_str.replace('-', ' ')
    sentence_clean_str = sentence_clean_str.replace('*', ' ')
    sentence_clean_str = sentence_clean_str.replace('&', ' ')
    # sentence_clean_str = sentence_clean_str.replace('/', ' ')
    sentence_clean_str = sentence_clean_str.replace('=', ' ')
    return sentence_clean_str.strip(string.punctuation)

def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    text = text.lower()
    word_list = nltk.word_tokenize(text)
    return [x for x in word_list if len(x) > min_word_return_size and x != '' and not is_number(x)]

'''
def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words
'''

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
    sentences = sentence_delimiters.split(text)
    return sentences


def build_stop_word_regex(stop_word_list):
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = '\\b' + word + '\\b'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


#
# Function that extracts the adjoined candidates from a list of sentences and filters them by frequency
#
def extract_adjoined_candidates(sentence_list, stoplist, min_keywords, max_keywords, min_freq):
    adjoined_candidates = []
    for s in sentence_list:
        # Extracts the candidates from each single sentence and adds them to the list
        adjoined_candidates += adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords)
    # Filters the candidates and returns them
    return filter_adjoined_candidates(adjoined_candidates, min_freq)


# return adjoined_candidates

#
# Function that extracts the adjoined candidates from a single sentence
#
def adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords):
    # Initializes the candidate list to empty
    candidates = []
    # Splits the sentence to get a list of lowercase words
    sl = s.lower().split()
    # For each possible length of the adjoined candidate
    for num_keywords in range(min_keywords, max_keywords + 1):
        # Until the third-last word
        for i in range(0, len(sl) - num_keywords):
            # Position i marks the first word of the candidate. Proceeds only if it's not a stopword
            if sl[i] not in stoplist:
                candidate = sl[i]
                # Initializes j (the pointer to the next word) to 1
                j = 1
                # Initializes the word counter. This counts the non-stopwords words in the candidate
                keyword_counter = 1
                contains_stopword = False
                # Until the word count reaches the maximum number of keywords or the end is reached
                while keyword_counter < num_keywords and i + j < len(sl):
                    # Adds the next word to the candidate
                    candidate = candidate + ' ' + sl[i + j]
                    # If it's not a stopword, increase the word counter. If it is, turn on the flag
                    if sl[i + j] not in stoplist:
                        keyword_counter += 1
                    else:
                        keyword_counter += 1
                        contains_stopword = True
                    # Next position
                    j += 1
                # Adds the candidate to the list only if:
                # 1) it contains at least a stopword (if it doesn't it's already been considered)
                # AND
                # 2) the last word is not a stopword
                # AND
                # 3) the adjoined candidate keyphrase contains exactly the correct number of keywords (to avoid doubles)
                if not [x for x in candidate.split() if len(x) >10]:
                    if contains_stopword and candidate.split()[-1] not in stoplist and keyword_counter == num_keywords and ' pron' not in candidate and 'pron ' not in candidate and ' pron ' not in candidate:
                        candidates.append(candidate)
    return candidates


#
# Function that filters the adjoined candidates to keep only those that appears with a certain frequency
#
def filter_adjoined_candidates(candidates, min_freq):
    # Creates a dictionary where the key is the candidate and the value is the frequency of the candidate
    candidates_freq = Counter(candidates)
    filtered_candidates = []
    # Uses the dictionary to filter the candidates
    for candidate in candidates:
        freq = candidates_freq[candidate]
        if freq >= min_freq:
            filtered_candidates.append(candidate)
    return filtered_candidates


def generate_candidate_keywords(sentence_list, stopword_pattern, stop_word_list, min_char_length=1, max_words_length=5,
                                min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "" and is_acceptable(phrase, min_char_length, max_words_length):
                phrase_list.append(phrase)
    phrase_list += extract_adjoined_candidates(sentence_list, stop_word_list, min_words_length_adj,
                                               max_words_length_adj, min_phrase_freq_adj)
    return phrase_list


def is_acceptable(phrase, min_char_length, max_words_length):
    if [x for x in phrase.split() if len(x) >10]:
        return 0

    if ' pron' in phrase or 'pron ' in phrase or ' pron ' in phrase:
        return 0

    # a phrase must have a min length in characters
    if len(phrase) < min_char_length:
        return 0   
    # a phrase must have a max number of words
    words = phrase.split()

    if len(words) > max_words_length:
        return 0

    digits = 0
    alpha = 0
    for i in range(0, len(phrase)):
        if phrase[i].isdigit():
            digits += 1
        elif phrase[i].isalpha():
            alpha += 1

    # a phrase must have at least one alpha character
    if alpha == 0:
        return 0

    # a phrase must have more alpha than digits characters
    if digits > alpha:
        return 0
    return 1


def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            #word_degree[word] += word_list_degree  # orig.
            word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        #word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
        word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score

def get_word_list(phrase_list):
    word_list = []
    for item in phrase_list:
        word_list += item.split()
    return word_list

def generate_candidate_keyword_scores(phrase_list, word_score, stop_word_list, min_keyword_frequency=1):
    keyword_candidates = {}
    words_dic = get_word_list(phrase_list)
    for phrase in phrase_list:
        #每个feature至少出现两次
        # if min_keyword_frequency > 1:
        #     if phrase_list.count(phrase) < min_keyword_frequency:
        #         continue

        #每个feature中的每个单词，都不是只出现一次
        #if min_keyword_frequency > 1:
        flag = True
        for item in phrase.split():
            if words_dic.count(item) >= min_keyword_frequency:
                flag = False
                break
        if flag:
            continue

        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        word_list = [x for x in word_list if x not in stop_word_list]
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


def get_business_name(business_id):
    with codecs.open(BUSINESS_FILE_PATH, encoding='utf-8') as f:
        for business_json in f:
            business = json.loads(business_json)  #convert json to dict
            if business['business_id'] != business_id:
                continue
            else:
                return business['name']

def get_user_name(user_id):
    with codecs.open(USER_FILE_PATH, encoding='utf-8') as f:
        for user_json in f:
            user = json.loads(user_json)  #convert json to dict
            if user['user_id'] != user_id:
                continue
            else:
                return user['name']
    
def feature_pruning_and_sentiment_analysis(feature_list, REVIEW_PATH_WITH_ID):
    message = []
    #排除含有非法字符的feature
    #排除/ 不在中间的feature
    list_one = []
    for item in feature_list:
        prog = re.compile("[^a-zA-Z\d\s+\$\￥\/]")
        result = prog.search(item[0])
        if result:
            # print('排除')
            continue
        elif len(item[0].split()) == 3 and '/' in item[0].split():
                if item[0].split()[1] != '/':
                    continue
                else:
                    list_one.append(item)
        else:
            # list_one.append(item)
            if '/' in item[0].split() and len(item[0].split()) < 3:
                continue
            else:
                list_one.append(item)
    
    #定位原句子
    #获得business_id, business_name,user_id, user_name, sentence，原句子中的feature
    #这里句子应该还需要经过相同的处理 才能找到

    #business_id已知

    list_two = []
    with codecs.open(REVIEW_PATH_WITH_ID, encoding='utf-8') as f:
        item = f.readline()
        W2 = ' 【business_id】:'
        W3 = ' 【content】:'
        pat2 = re.compile(W2+'(.*?)'+W3, re.S)
        business_id = pat2.findall(item)[0]
        business_name = get_business_name(business_id)
        business_name = business_name.lower()
        business_name_list = business_name.split()
        for feature in list_one:
            flag = False
            feature_ori = feature
            feature_chan = feature[0]
            feature_chan = ' '.join(feature_chan.split())
            
            #先取出user_id，business_id 和 content
                
            for i in feature_chan.split():
                if i in business_name_list:
                    print('特征包含商家名 business_name:',business_name,'   排除feature: ',feature_chan)
                    #这个feature被排除
                    flag = True
                    break

            if not flag:
                list_two.append(feature_ori)

    list_two = list_two[0:(len(list_two) // 3)]
    
    print(len(list_two))

    list_three = []
    for item in list_two:
        feature = item[0]
        feature_list = feature.split()
        
        flag = False
        for i in feature_list:
            tagged = nltk.pos_tag([i])[0][1]
            if tagged == 'VB':
                flag = True
                break
        if flag:
            continue

        if len(feature_list) == 1:
            tagged = nltk.pos_tag([feature_list[0]])[0][1]
            if tagged == 'JJ' or tagged == 'JJR' or tagged == 'JJS' or tagged =='JJR' or tagged == 'RB' or tagged == 'UH' or tagged == 'VB':
                print('排除feature: ', feature)
            else:
                print(feature)
                list_three.append(item)
        else:c
            list_three.append(item)

    print(len(list_three))


    with codecs.open(REVIEW_PATH_WITH_ID, encoding='utf-8') as f:
        item = f.readline()
        W1 = '【user_id】:'
        W2 = ' 【business_id】:'
        W3 = ' 【content】:'
        pat2 = re.compile(W2+'(.*?)'+W3, re.S)
        business_id = pat2.findall(item)[0]
        business_name = get_business_name(business_id)

        
        muti_para = []
        for item in f:
            content = []
            content.append(item)
            content.append(list_three)
            content.append(business_id)
            content.append(business_name)
            muti_para.append(content)


        #启动并行计算
        #cores = multiprocessing.cpu_count()
        #pool = Pool(cores)
        pool = Pool()
        cnt = 1
        for y in pool.imap(sentiment_analysis, muti_para):
            message += y
            cnt += 1
            print('正在处理第 {} 个review'.format(cnt))
        pool.close()
        pool.join()

    print('最终提取出 {} 个feature'.format(len(message)))
    return message


def sentiment_analysis(content_list):
    item = content_list[0]
    list_three = content_list[1]
    business_id = content_list[2]
    business_name = content_list[3]
    message = []

    #先取出user_id，和 content
    W1 = '【user_id】:'
    W2 = ' 【business_id】:'
    W3 = ' 【content】:'
    pat1 = re.compile(W1+'(.*?)'+W2, re.S)
    user_id = pat1.findall(item)[0]
    pat3 = re.compile(W3+'(.*?)$', re.S)
    content = pat3.findall(item)[0]
    user_name = get_user_name(user_id)


    sentence_list = nltk.sent_tokenize(content)   

    for sentence in sentence_list:


        sentence_orig = sentence.strip(string.punctuation)
        sentence_clean = text_clean(sentence)
        
        #可选
        sentence_clean = ' '.join(sentence_clean.split())

        for feature in list_three:
            # feature: ('paolos uye photowalk', 8.5)
            # 先转化为 'paolos uye photowalk'
            feature_orig = feature
            feature_cha = feature[0]
            check_mad = True
            
            #可选
            feature_cha = ' '.join(feature_cha.split())

            if feature_cha in sentence_clean:
                #该feature在这个句子


                #如果 原句子中的feature 每个单词首字母都是大写，那么很可能就是地名，是该商家附近的地名， 不能当做feature但是有价值，  总之先把他弄出来
                # 可选，因为会耗时
                feature_cha_list = feature_cha.split()
                sentence_clean_list = sentence_clean.split()
                sentence_orig_list = sentence_orig.split()
                word_temp = ""

                
                try:
                    if len(feature_cha_list) == 2 and feature_cha_list[0] in sentence_clean_list and feature_cha_list[1] in sentence_clean_list:
                        word1 = sentence_orig_list[sentence_clean_list.index(feature_cha_list[0])]
                        word2 = sentence_orig_list[sentence_clean_list.index(feature_cha_list[1])]
                        if word1[0].isupper() and word2[0].isupper():
                            check_mad = False
                            word_temp = word1 + ' ' + word2

                    if len(feature_cha_list) == 3 and feature_cha_list[0] in sentence_clean_list and feature_cha_list[1] in sentence_clean_list and feature_cha_list[2] in sentence_clean_list: 
                        word1 = sentence_orig_list[sentence_clean_list.index(feature_cha_list[0])]
                        word2 = sentence_orig_list[sentence_clean_list.index(feature_cha_list[1])]
                        word3 = sentence_orig_list[sentence_clean_list.index(feature_cha_list[2])]
                        if word1[0].isupper() and word2[0].isupper() and word3[0].isupper():
                            check_mad = False
                            word_temp = word1 + ' ' + word2 + ' ' + word3


                    if not check_mad:
                        #命名实体识别 label：GPE，PERSON
                        ne_str = ""
                        doc = nlp(sentence_orig)
                        for ent in doc.ents:
                            if ent.label_ == 'PERSON' or ent.label_ == 'GPE':
                                # print(ent.text, ent.label_)
                                ne_str = ne_str + ' ' + ent.text

                        if word_temp in ne_str:
                            print('特殊feature：', word_temp)
                            list_temp.append(word_temp)
                            continue 

                except IndexError as e:
                    print(e)
                
                ##############

                #feature做替换

                if '$' in feature_cha or '￥' in feature_cha:
                    feature_cha = 'price'
                if 'staff' in feature_cha or 'waiter' in feature_cha or 'waitress' in feature_cha or 'employee' in feature_cha:
                    feature_cha = 'service'

                have_noun = False
                #feature_cha_list = feature_cha.split()
                tagged = nltk.pos_tag(feature_cha_list)
                for item in tagged:
                    if item[1] == 'NN' or item[1] == 'NNS' or item[1] == 'NNP' or item[1] == 'NNPS':
                        have_noun = True
                        break
                if not have_noun:
                    print('feature中没有名词：', feature_cha) 
                    continue
                #剩下的都是合理的feature，进行情感分析

                polarity,subjectivity = TextBlob(sentence).sentiment
                polarity = round(polarity, 2)
                subjectivity = round(subjectivity, 2)

                part = {}
                part['business_id'] = business_id
                part['business_name'] = business_name
                part['user_id'] = user_id
                part['user_name'] = user_name
                part['feature'] = feature_cha
                part['polarity'] = polarity
                part['subjectivity'] = subjectivity
                message.append(part)
            else:
                continue
    return message
def write_feature_into_file(message):
    with codecs.open(FEATURE_MESSAGE, 'a', encoding='utf-8') as f:
        for item in message:
            jsObj = json.dumps(item, sort_keys=True, separators=(',', ': '))
            f.write(jsObj)
            f.write('\n')

class Rake(object):

    '''
    stop_words_path:停用词路径
    min_char_length:feature字符至少的长度
    max_words_length:feature最多包含的单词数
    min_keyword_frequency:关键词（feature的成员）出现的频率
    min_words_length_adj:邻接feature最少单词数
    max_words_length_adj:邻接feature最多单词数
    min_phrase_freq_adj：邻接feature在文本中至少出现的次数
    '''
    def __init__(self, stop_words_path, min_char_length=1, max_words_length=5, min_keyword_frequency=1,
                 min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2):
        self.__stop_words_path = stop_words_path
        self.__stop_words_list = load_stop_words(stop_words_path)
        self.__min_char_length = min_char_length
        self.__max_words_length = max_words_length
        self.__min_keyword_frequency = min_keyword_frequency
        self.__min_words_length_adj = min_words_length_adj
        self.__max_words_length_adj = max_words_length_adj
        self.__min_phrase_freq_adj = min_phrase_freq_adj

    def run(self, text):
        sentence_list = split_sentences(text)
        stop_words_pattern = build_stop_word_regex(self.__stop_words_list)
        phrase_list = generate_candidate_keywords(sentence_list, stop_words_pattern, self.__stop_words_list,
                                                  self.__min_char_length, self.__max_words_length,
                                                  self.__min_words_length_adj, self.__max_words_length_adj,
                                                  self.__min_phrase_freq_adj)
        # print(get_word_list(phrase_list))

        word_scores = calculate_word_scores(phrase_list)
        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores, self.__stop_words_list, self.__min_keyword_frequency)
        sorted_keywords = sorted(six.iteritems(keyword_candidates), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords


if test:
    # text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    # print(sortedKeywords[0:(totalKeywords // 3)])
    
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
        # print(keywords)
        #print(len(keywords))
        #print(keywords[0:(len(keywords) // 3)])
        message = feature_pruning_and_sentiment_analysis(keywords, REVIEW_PATH_WITH_ID)

        write_feature_into_file(message)
