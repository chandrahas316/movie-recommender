import time    #required libraries
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pickle

class preprocessing:
    # tokenization,stop words,stemming
    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()
        self.stopwords_remove = stopwords.words('english')
        self.stemmer = PorterStemmer()

    # reading csv file and storing it in a dataframe
    def read_dataset(self):
        dataset = pd.read_csv('movie_dataset.csv')
        f_handler = open("movie_plot.obj","wb")
        pickle.dump(dataset,f_handler)
        f_handler.close()
        return dataset

    # converts words to lower case and replaces NA with ''
    def to_lowercase(self, dataset):
        dataset = dataset.fillna('')
        attr = ['Plot','Title','Origin/Ethnicity', 'Director', 'Cast', 'Genre' ]
        for i in attr:
            dataset[i] = dataset[i].str.lower()
        dataset = dataset.fillna('')
        return dataset

    # removes punctuations and escape sequences
    def data_peprocess(self, words):
        words = words.replace("\n"," ").replace("\r"," ")
        words = words.replace("'s"," ")
        punct_list = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
        x = str.maketrans(dict.fromkeys(punct_list," "))
        words = words.translate(x)
        return words

    # tokenizes the words
    def tokhel(self, words):
        words = self.data_peprocess(words)
        return self.tokenizer.tokenize(words)

    # tokenizing words in dataset columns
    def tokenization(self, dataset):
        dataset['TokensPlot'] = dataset['Plot'].apply(self.tokhel)
        dataset['TokensTitle'] = dataset['Title'].apply(self.tokhel)
        dataset['TokensOrigin'] = dataset['Origin/Ethnicity'].apply(self.tokhel)
        dataset['TokensDirector'] = dataset['Director'].apply(self.tokhel)
        dataset['TokensCast'] = dataset['Cast'].apply(self.tokhel)
        dataset['TokensGenre'] = dataset['Genre'].apply(self.tokhel)
        dataset['Tokens'] = dataset['TokensPlot'] + dataset['TokensTitle'] + dataset['TokensOrigin'] + dataset['TokensDirector'] + dataset['TokensCast'] + dataset['TokensGenre']
        dataset['Length'] = dataset.Tokens.apply(len)
        dataset['TitleLength'] = dataset.TokensTitle.apply(len)
        return dataset

    stop_words = ['0o', '0s', '3a', '3b', '3d', '6b', '6o', 'a', 'a1', 'a2', 'a3', 'a4', 'ab', 'able', 'about', 'above',
              'abst', 'ac', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'ad', 'added', 'adj',
              'ae', 'af', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'ag', 'again', 'against', 'ah',
              'ain', "ain't", 'aj', 'al', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also',
              'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another',
              'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ao', 'ap',
              'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'ar', 'are', 'aren',
              'arent', "aren't", 'arise', 'around', 'as', "a's", 'aside', 'ask', 'asking', 'associated', 'at', 'au',
              'auth', 'av', 'available', 'aw', 'away', 'awfully', 'ax', 'ay', 'az', 'b', 'b1', 'b2', 'b3', 'ba', 'back',
              'bc', 'bd', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',
              'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'believe', 'below', 'beside', 'besides',
              'best', 'better', 'between', 'beyond', 'bi', 'bill', 'biol', 'bj', 'bk', 'bl', 'bn', 'both', 'bottom',
              'bp', 'br', 'brief', 'briefly', 'bs', 'bt', 'bu', 'but', 'bx', 'by', 'c', 'c1', 'c2', 'c3', 'ca', 'call',
              'came', 'can', 'cannot', 'cant', "can't", 'cause', 'causes', 'cc', 'cd', 'ce', 'certain', 'certainly',
              'cf', 'cg', 'ch', 'changes', 'ci', 'cit', 'cj', 'cl', 'clearly', 'cm', "c'mon", 'cn', 'co', 'com', 'come',
              'comes', 'con', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing',
              'contains', 'corresponding', 'could', 'couldn', 'couldnt', "couldn't", 'course', 'cp', 'cq', 'cr',
              'cry', 'cs', "c's", 'ct', 'cu', 'currently', 'cv', 'cx', 'cy', 'cz', 'd', 'd2', 'da', 'date', 'dc', 'dd',
              'de', 'definitely', 'describe', 'described', 'despite', 'detail', 'df', 'di', 'did', 'didn', "didn't",
              'different', 'dj', 'dk', 'dl', 'do', 'does', 'doesn', "doesn't", 'doing', 'don', 'done', "don't", 'down',
              'downwards', 'dp', 'dr', 'ds', 'dt', 'du', 'due', 'during', 'dx', 'dy', 'e', 'e2', 'e3', 'ea', 'each',
              'ec', 'ed', 'edu', 'ee', 'ef', 'effect', 'eg', 'ei', 'eight', 'eighty', 'either', 'ej', 'el', 'eleven',
              'else', 'elsewhere', 'em', 'empty', 'en', 'end', 'ending', 'enough', 'entirely', 'eo', 'ep', 'eq', 'er',
              'es', 'especially', 'est', 'et', 'et-al', 'etc', 'eu', 'ev', 'even', 'ever', 'every', 'everybody',
              'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'ey', 'f', 'f2', 'fa',
              'far', 'fc', 'few', 'ff', 'fi', 'fifteen', 'fifth', 'fify', 'fill', 'find', 'fire', 'first', 'five',
              'fix', 'fj', 'fl', 'fn', 'fo', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth',
              'forty', 'found', 'four', 'fr', 'from', 'front', 'fs', 'ft', 'fu', 'full', 'further', 'furthermore',
              'fy', 'g', 'ga', 'gave', 'ge', 'get', 'gets', 'getting', 'gi', 'give', 'given', 'gives', 'giving', 'gj',
              'gl', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'gr', 'greetings', 'gs', 'gy', 'h', 'h2', 'h3',
              'had', 'hadn', "hadn't", 'happens', 'hardly', 'has', 'hasn', 'hasnt', "hasn't", 'have', 'haven',
              "haven't", 'having', 'he', 'hed', "he'd", "he'll", 'hello', 'help', 'hence', 'her', 'here', 'hereafter',
              'hereby', 'herein', 'heres', "here's", 'hereupon', 'hers', 'herself', 'hes', "he's", 'hh', 'hi', 'hid',
              'him', 'himself', 'his', 'hither', 'hj', 'ho', 'home', 'hopefully', 'how', 'howbeit', 'however', "how's",
              'hr', 'hs', 'http', 'hu', 'hundred', 'hy', 'i', 'i2', 'i3', 'i4', 'i6', 'i7', 'i8', 'ia', 'ib', 'ibid',
              'ic', 'id', "i'd", 'ie', 'if', 'ig', 'ignored', 'ih', 'ii', 'ij', 'il', "i'll", 'im', "i'm", 'immediate',
              'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'indeed', 'index', 'indicate',
              'indicated', 'indicates', 'information', 'inner', 'insofar', 'instead', 'interest', 'into', 'invention',
              'inward', 'io', 'ip', 'iq', 'ir', 'is', 'isn', "isn't", 'it', 'itd', "it'd", "it'll", 'its', "it's",
              'itself', 'iv', "i've", 'ix', 'iy', 'iz', 'j', 'jj', 'jr', 'js', 'jt', 'ju', 'just', 'k', 'ke', 'keep',
              'keeps', 'kept', 'kg', 'kj', 'km', 'know', 'known', 'knows', 'ko', 'l', 'l2', 'la', 'largely', 'last',
              'lately', 'later', 'latter', 'latterly', 'lb', 'lc', 'le', 'least', 'les', 'less', 'lest', 'let', 'lets',
              "let's", 'lf', 'like', 'liked', 'likely', 'line', 'little', 'lj', 'll', 'll', 'ln', 'lo', 'look',
              'looking', 'looks', 'los', 'lr', 'ls', 'lt', 'ltd', 'm', 'm2', 'ma', 'made', 'mainly', 'make', 'makes',
              'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg', 'might',
              'mightn', "mightn't", 'mill', 'million', 'mine', 'miss', 'ml', 'mn', 'mo', 'more', 'moreover', 'most',
              'mostly', 'move', 'mr', 'mrs', 'ms', 'mt', 'mu', 'much', 'mug', 'must', 'mustn', "mustn't", 'my',
              'myself', 'n', 'n2', 'na', 'name', 'namely', 'nay', 'nc', 'nd', 'ne', 'near', 'nearly', 'necessarily',
              'necessary', 'need', 'needn', "needn't", 'needs', 'neither', 'never', 'nevertheless', 'new', 'next',
              'ng', 'ni', 'nine', 'ninety', 'nj', 'nl', 'nn', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone',
              'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'novel', 'now', 'nowhere', 'nr', 'ns', 'nt', 'ny',
              'o', 'oa', 'ob', 'obtain', 'obtained', 'obviously', 'oc', 'od', 'of', 'off', 'often', 'og', 'oh', 'oi',
              'oj', 'ok', 'okay', 'ol', 'old', 'om', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'oo',
              'op', 'oq', 'or', 'ord', 'os', 'ot', 'other', 'others', 'otherwise', 'ou', 'ought', 'our', 'ours',
              'ourselves', 'out', 'outside', 'over', 'overall', 'ow', 'owing', 'own', 'ox', 'oz', 'p', 'p1', 'p2',
              'p3', 'page', 'pagecount', 'pages', 'par', 'part', 'particular', 'particularly', 'pas', 'past', 'pc',
              'pd', 'pe', 'per', 'perhaps', 'pf', 'ph', 'pi', 'pj', 'pk', 'pl', 'placed', 'please', 'plus', 'pm', 'pn',
              'po', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'pq', 'pr', 'predominantly', 'present',
              'presumably', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'ps', 'pt', 'pu',
              'put', 'py', 'q', 'qj', 'qu', 'que', 'quickly', 'quite', 'qv', 'r', 'r2', 'ra', 'ran', 'rather', 'rc',
              'rd', 're', 'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs', 'regarding',
              'regardless', 'regards', 'related', 'relatively', 'research', 'research-articl', 'respectively',
              'resulted', 'resulting', 'results', 'rf', 'rh', 'ri', 'right', 'rj', 'rl', 'rm', 'rn', 'ro', 'rq', 'rr',
              'rs', 'rt', 'ru', 'run', 'rv', 'ry', 's', 's2', 'sa', 'said', 'same', 'saw', 'say', 'saying', 'says',
              'sc', 'sd', 'se', 'sec', 'second', 'secondly', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming',
              'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'sf',
              'shall', 'shan', "shan't", 'she', 'shed', "she'd", "she'll", 'shes', "she's", 'should', 'shouldn',
              "shouldn't", "should've", 'show', 'showed', 'shown', 'showns', 'shows', 'si', 'side', 'significant',
              'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty', 'sj', 'sl', 'slightly',
              'sm', 'sn', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime',
              'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'sp', 'specifically', 'specified', 'specify',
              'specifying', 'sq', 'sr', 'ss', 'st', 'still', 'stop', 'strongly', 'sub', 'substantially',
              'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure', 'sy', 'system', 'sz', 't', 't1', 't2',
              't3', 'take', 'taken', 'taking', 'tb', 'tc', 'td', 'te', 'tell', 'ten', 'tends', 'tf', 'th', 'than',
              'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that's", "that've", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'thered', 'therefore',
              'therein', "there'll", 'thereof', 'therere', 'theres', "there's", 'thereto', 'thereupon', "there've",
              'these', 'they', 'theyd', "they'd", "they'll", 'theyre', "they're", "they've", 'thickv', 'thin', 'think',
              'third', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thousand', 'three',
              'throug', 'through', 'throughout', 'thru', 'thus', 'ti', 'til', 'tip', 'tj', 'tl', 'tm', 'tn', 'to',
              'together', 'too', 'took', 'top', 'toward', 'towards', 'tp', 'tq', 'tr', 'tried', 'tries', 'truly',
              'try', 'trying', 'ts', "t's", 'tt', 'tv', 'twelve', 'twenty', 'twice', 'two', 'tx', 'u', 'u201d', 'ue',
              'ui', 'uj', 'uk', 'um', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto',
              'uo', 'up', 'upon', 'ups', 'ur', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses',
              'using', 'usually', 'ut', 'v', 'va', 'value', 'various', 'vd', 've', 've', 'very', 'via', 'viz', 'vj',
              'vo', 'vol', 'vols', 'volumtype', 'vq', 'vs', 'vt', 'vu', 'w', 'wa', 'want', 'wants', 'was', 'wasn',
              'wasnt', "wasn't", 'way', 'we', 'wed', "we'd", 'welcome', 'well', "we'll", 'well-b', 'went', 'were',
              "we're", 'weren', 'werent', "weren't", "we've", 'what', 'whatever', "what'll", 'whats', "what's", 'when',
              'whence', 'whenever', "when's", 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres',
              "where's", 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 'who', 'whod',
              'whoever', 'whole', "who'll", 'whom', 'whomever', 'whos', "who's", 'whose', 'why', "why's", 'wi',
              'widely', 'will', 'willing', 'wish', 'with', 'within', 'without', 'wo', 'won', 'wonder', 'wont',
              "won't", 'words', 'world', 'would', 'wouldn', 'wouldnt', "wouldn't", 'www', 'x', 'x1', 'x2', 'x3', 'xf',
              'xi', 'xj', 'xk', 'xl', 'xn', 'xo', 'xs', 'xt', 'xv', 'xx', 'y', 'y2', 'yes', 'yet', 'yj', 'yl', 'you',
              'youd', "you'd", "you'll", 'your', 'youre', "you're", 'yours', 'yourself', 'yourselves', "you've", 'yr',
              'ys', 'yt', 'z', 'zero', 'zi', 'zz'
            ]


    # removes stop words from dataset columns
    def stopwords_removal(self, dataset):
        dataset['Tokens'] = dataset['Tokens'].apply(lambda x: [i for i in x if i not in self.stopwords_remove])
        return dataset

    # stemming
    def data_stemming(self, dataset, x):
        dataset['stemmed'] = dataset[x].apply(lambda x: [self.stemmer.stem(word) for word in x])
        return dataset

    # creating a dict with keys as words and values as frequency
    def words_dictionary(self, uni, tokens):
        uniq = tuple(uni)
        word_dict = dict.fromkeys(uniq, 0)
        for w in tokens:
            word_dict[w] += 1
        return word_dict

    # calculating term frequency
    def term_freq(self, dataset_tknzd):
        dataset_tknzd['uni_w'] = dataset_tknzd['stemmed'].apply(set)
        dataset_tknzd['Frequency'] = dataset_tknzd.apply(lambda x: self.words_dictionary(x.uni_w, x.stemmed), axis=1)
        return dataset_tknzd

    # creating inverted index
    def create_inv_index(self, dataset_tknzd):
        Inverted_Index = pd.DataFrame()
        tokens = set(dataset_tknzd['uni_w'][0])
        for i in range (0, 34885):
            tokens = set.union(tokens,set(dataset_tknzd['uni_w'][i+1]))
        Inverted_Index = pd.DataFrame(tokens)
        Inverted_Index.columns =['Words']
        return Inverted_Index

    # adding posting lists to inverted index
    def posting_lists(self, Inverted_Index, dataset_tknzd):
        inverted_index_dict = {}
        for i in range (0, 34886):
            for item in dataset_tknzd['uni_w'][i]:
                if item in inverted_index_dict.keys():
                    inverted_index_dict[item]+=1
                else:
                    inverted_index_dict[item]=1
           
        Inverted_Index = pd.Series(inverted_index_dict).to_frame()

        Inverted_Index.columns =['PostingList']
        return Inverted_Index

    def write_to_pickle(self, file, dataset): #storing the processed data in a pickle file.
        f_handler = open(file,"wb")
        pickle.dump(dataset,f_handler)
        f_handler.close()




    def main(self):
        dataset = self.read_dataset()
        dataset = self.to_lowercase(dataset)
        dataset = self.tokenization(dataset)
        dataset = self.stopwords_removal(dataset)
        dataset = self.data_stemming(dataset, 'Tokens')
        dataset = self.term_freq(dataset)
        
        dataset1 = dataset[['Length' , 'Frequency']]
        self.write_to_pickle("processed_data.obj", dataset1)
        
        dataset_Title = dataset[['TokensTitle', 'TitleLength']]
        dataset_Title = self.data_stemming(dataset_Title, 'TokensTitle')

        dataset_Title = self.term_freq(dataset_Title)
        self.write_to_pickle("processed_data_title.obj", dataset_Title)

        Inverted_Index = self.create_inv_index(dataset)
        Inverted_Index = self.posting_lists(Inverted_Index, dataset)
        self.write_to_pickle("inverted_index.obj", Inverted_Index)

        Inverted_Index_Title = self.create_inv_index(dataset_Title)
        Inverted_Index_Title = self.posting_lists(Inverted_Index_Title, dataset_Title)
        self.write_to_pickle("inverted_index_title.obj", Inverted_Index_Title)

        # tf_idf_dict = self.tf_idf("processed_data.obj", "inverted_index.obj", "Length")
        # filehandler = open("tf-idf.obj", "wb")
        # pickle.dump(tf_idf_dict, filehandler)
        # filehandler.close()
        #
        # tf_idf_title_dict = self.tf_idf("processed_data_title.obj", "inverted_index_title.obj", "TitleLength")
        # filehandler = open("tf-idf_title.obj", "wb")
        # pickle.dump(tf_idf_title_dict, filehandler)
        # filehandler.close()
print("This process may take about 8 to 10 minutes to complete...SO BE PATIENT")
Data = preprocessing()
Data.main()

