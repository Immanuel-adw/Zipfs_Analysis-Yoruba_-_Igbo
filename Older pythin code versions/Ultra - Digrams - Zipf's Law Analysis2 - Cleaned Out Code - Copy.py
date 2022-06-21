# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 04:40:12 2020import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

@author: imman
"""
import nltk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as stats
import statistics
import pandas as pd
import re
import seaborn as sns
from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
from math import log
from collections import Counter
from nltk.book import*
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.util import ngrams
import nltk.corpus as corpus
from scipy.stats import kstest
from functools import reduce
#%matplotlib inline





# function for taking corpus key-word and returning the list-of-words and Title

def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data),num)
    return [' '.join(grams) for grams in n_grams]


def takecorp(corp):
    all_words_clean = read_docu(corp)
    if n_gram_value > 0:
        if n_gram_value == 1:
            name = 'Unigrams - '
        elif n_gram_value == 2:
            name = 'Digrams - '
        elif n_gram_value == 3:
            name = 'Trigrams - '
    elif n_gram_value == 0:
            name = ''
    if corp == igbb:
        Title = name + 'Igbo Bible'
    elif corp == YCB:
        Title = name + 'Yoruba Bible'
    elif corp == WEB:
        Title = name + 'World English Bible'
    elif corp == b:
        Title = name + 'Brown Corpus'
    elif corp == kjv:
        Title = name + 'King James Version'
    elif corp == bbcYr:
        Title = name + 'BBC Yoruba Corpus'
    elif corp == bbcigb:
        Title = name + 'BBC Igbo Corpus'
    elif corp == reut:
        Title = name + 'Reuters Corpus'
    
    return [all_words_clean, Title]



# In[1]:

#Best fit Zipf-Mandelbrot Hypothesis Trial Equation 
def best_fit(x, m, b, c):
    return m*np.log(x+b) + c 

    
def tokencount_aftercutoff(word_counts_n):
    return token_counts(word_counts_n)[1]


#Phython Function that reads the different files and filetypes giving out the full word-list
def read_docu(file):
    all_words = []
    
    if type(file) == nltk.text.Text:
        text = [str(file)]
        chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
        tr = str.maketrans("", "", chars_to_remove) 
        text = [text[0].translate(tr)]
        if n_gram_value > 0:
            all_words = extract_ngrams(str(text), n_gram_value)
        elif n_gram_value == 0:
            for line in text:
                line = line.lower()
                line = line.strip().split()
                all_words += line
        return(all_words)
        
    elif type(file) == NLTKClass1 or type(file) == NLTKClass2 or type(file)== NLTKClass3:
        chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
        text = []
        for word in file.words():
            text.append(word)
        text = [str(text)]
        tr = str.maketrans("", "", chars_to_remove) 
        text = [text[0].translate(tr)]
        if n_gram_value > 0:
            all_words = extract_ngrams(str(text), n_gram_value)
        elif n_gram_value == 0:
            for line in text:
                line = line.lower()
                line = line.strip().split()
                all_words += line
        return(all_words)
        
    else:
        with open(file, "r", encoding = "utf-8") as input_file:
            text = [input_file.read()]
            chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
            tr = str.maketrans("", "", chars_to_remove) 
            text = [text[0].translate(tr)]
            if n_gram_value > 0:
                all_words = extract_ngrams(str(text), n_gram_value)
            elif n_gram_value == 0:
                for line in text:
                    line = line.lower()
                    line = line.strip().split()
                    all_words += line
            return(all_words)  

             
def corp_df_aftercut_off(all_words_clean, cutoff_freq):
    
    #Code to turn word-list into Data-Frame with rank, frequency and their logs
    
    word_list = all_words_clean
    words = pd.DataFrame(word_list, columns=['word'])
    word_counts1 = words.word.value_counts().reset_index()
    word_counts1.columns = ['word', 'n_raw']
    word_counts1['word_rank_raw'] = word_counts1.n_raw.rank(ascending=False)
    #i = word_counts['word_rank']
    #word_counts['log_n'] = [log(i,np.exp(1)) for i in word_counts.n]
    #word_counts['log_rank'] = [log(i,np.exp(1)) for i in word_counts.word_rank]
    
    if cutoff_freq > 0:
        n_raw = word_counts1.n_raw.tolist()
        #word_rank_raw = word_counts1.word_rank_raw.tolist()
        cut_off_index = n_raw.index(cutoff_freq)
        word_counts2 = word_counts1.iloc[:cut_off_index, :]
        word_counts2.columns = ['word2', 'n2', 'word_rank2']
        #words_trimmed = words.word[:cut_off_index].tolist()
    elif cutoff_freq == 0:
        word_counts2 = word_counts1
        word_counts2.columns = ['word2', 'n2', 'word_rank2']
        #word_counts2['n2'] = word_counts2.n_raw[:]
        #word_counts2['word_rank2'] = word_counts1.word_rank_raw[:]
        #words_trimmed = words.word.tolist()
    return word_counts2


def token_counts(word_counts_n):
    num_tokens = len(all_words_clean)
    num_types = len(set(all_words_clean))
    num_tokens_used = 0
    #print('word_counts[n] is ', word_counts2['n2'])
    for n in word_counts_n.tolist():
        num_tokens_used+=n
    #print('num_tokens_used is ', num_tokens_used)   
    #print('num of tokens from source corpus is ', num_tokens)
    print('No of tokens in '+ str(Title)+' is ', num_tokens)
    print('No of tokens used after Freq. cutoff from the '+ str(Title)+' for analysis is ', str(num_tokens_used))
    print('No of types in source corpus is ', num_types)
    return [num_tokens, num_tokens_used, num_types]


def bootstrap_dfs(all_words_clean, num_bootstraps, cutoff_freq):
    #Making Bootstraps
    word_counts2 = corp_df_aftercut_off(all_words_clean, cutoff_freq)
    df = word_counts2.iloc[:,:2]
    df.columns = ['bootword2', 'boot_n2']
    bootstraps_Dfs = [df]
    y_bootstraps = []
    boot_xs = []
    bootstraplists = []
    num_tokens = len(all_words_clean)
    for i in range(num_bootstraps):
        max_val = num_tokens
        size_of_corpus = num_tokens
        rand_word_indices = np.random.randint(0, max_val, size_of_corpus) # an array of random indices 
        arr_text = np.array(all_words_clean) #need to use numpy as its waaaay quicker
        new_text = list(arr_text[rand_word_indices]) # quicker way to get the new corpus from the original corpus
        bootstrap_samp = new_text             
        bootwords = pd.DataFrame(bootstrap_samp, columns=['bootword'])
        boot_counts = bootwords.bootword.value_counts().reset_index()
        boot_counts.columns = ['boot_word', 'boot_n_raw']
        boot_counts['boot_rank_raw'] = boot_counts.boot_n_raw.rank(ascending=False)
        
        if cutoff_freq > 0:
            boot_n_raw = boot_counts.boot_n_raw.tolist()
            #boot_rank_raw = boot_counts.boot_rank_raw.tolist()
            cut_off_index = boot_n_raw.index(cutoff_freq)
            #boot_counts2 = boots_trimmed.bootword2.value_counts().reset_index()
            boot_counts2 = boot_counts.iloc[:cut_off_index,:]
            boot_counts2.columns = ['bootword2', 'boot_n2', 'boot_rank2']
            #words_trimmed = words.word[:cut_off_index].tolist()
        elif cutoff_freq == 0:
            boot_counts2 = boot_counts
            boot_counts2['boot_n2'] = boot_counts.boot_n_raw[:]
            boot_counts2['boot_rank2'] = boot_counts.boot_rank_raw[:]
            #words_trimmed = words.word.tolist()
        bootstraps_Dfs.append(boot_counts2.iloc[:,:2]) 
        bootrank = boot_counts2.boot_rank2
        bootFreq = boot_counts2.boot_n2
        yvalues = np.log(bootFreq)
        xvalues = bootrank
        bootstraplists.append(bootstrap_samp)
        boot_xs.append(xvalues.tolist())
        y_bootstraps.append(yvalues.tolist())
        
    bootstrap_Dataframes = reduce(lambda  left,right: pd.merge(left,right,on=['bootword2'], how="inner"), bootstraps_Dfs).fillna(0)
    return [bootstrap_Dataframes, boot_xs, y_bootstraps]


def y_uncert_hists(word_merged_Dataframe, n_most_freq_words, num_bins):
    for i in range(n_most_freq_words):
        word = word_merged_Dataframe.iloc[i,0]
        values = word_merged_Dataframe.iloc[i,1:].to_list()
        values = np.log(values) 
        plt.hist(values, bins = num_bins)
        plt.title('rank = '+ str(i + 1) +'| word = ' + word)
        plt.show()


# In[2]:

#NLTK filetype Classes
NLTKClass1 = nltk.corpus.reader.tagged.CategorizedTaggedCorpusReader
NLTKClass2 = nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader
NLTKClass3 = nltk.corpus.util.LazyCorpusLoader


# Corpus Keys

#Complete bibles Yoruba YCB, Igbo igbb, English WEB & kjv
WEB = r'WEB.txt'
YCB = r'YCB.txt'
igbb = r'igbobible.txt'
kjv = r'KJV Gutenberg.txt' 

#i = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Online Kaggle Zipf Testers\human_text.txt'
#j = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Online Kaggle Zipf Testers\robot_text.txt'
#YCB_OT = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\YCB OT.txt'
#YCB_NT = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\YCB NT.txt'
bbcYr = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Yoruba Bibeli Mimo\BBC Datasets\BBC Yoruba Articles\bbc yoruba raw1.txt'
#TED =  r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\TedTalks\TedTalks 2017.txt'
bbcigb = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\BBC Datasets\bbcigbo.txt'
b = brown
#wbtxt = webtext
#gut = gutenberg
reut = corpus.reuters
#inaug = corpus.inaugural 



# Setting key Values 
#
#
n_gram_value = 0        #ngram value, uni-gram n = 1, di-gram =2, tri-gram n = 3, n=0 uses written tokenizer as opposed to NLTK tokenizer (expected to yield identical results)
cutoff_freq = 5        #Cuts off words which occur at a frequency 10 and less
num_bootstraps = 100     #Gives the number of bootstraps which should be run
corp_key = igbb         #For picking the desired corpus using the keys above



# Calling takecorp() function, and defining all_words_clean word-list, to be called later
corp_words_and_Title = takecorp(corp_key)
all_words_clean = corp_words_and_Title[0]
Title = corp_words_and_Title[1]


# Calling Bootstrap DataFrame Function
bootstrap_Dfs = bootstrap_dfs(all_words_clean, num_bootstraps, cutoff_freq)
boot_Dfs = bootstrap_Dfs[0]
boot_xs = bootstrap_Dfs[1]
y_bootstraps = bootstrap_Dfs[2]

#print (bootstrap_Dfs)




'''
#For importing corpora combinations
all_words_clean1 = read_docu(wbtxt)
all_words_clean2 = read_docu(WEB)
all_words_clean3 = read_docu(b)
all_words_clean4 = read_docu(gut)

all_words_clean = all_words_clean1 + all_words_clean2 + all_words_clean3 + all_words_clean4

'''

word_counts2 = corp_df_aftercut_off(all_words_clean, cutoff_freq)

token_counts(word_counts2['n2'])

word_counts2['log_n'] = np.log(word_counts2.n2)
word_counts2['log_rank'] = np.log(word_counts2.word_rank2)



# Defining Plot Parameters - for trial function matplotlib.pyplot 
log_n = word_counts2['log_n']
log_rank = word_counts2['log_rank']
#print(x)
xrank = word_counts2.word_rank2
#print ('y si ', log_n)
#print(word_counts['n']
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 04:40:12 2020import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

@author: imman
"""
import nltk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as stats
import statistics
import pandas as pd
import re
import seaborn as sns
from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
from math import log
from collections import Counter
from nltk.book import *
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.util import ngrams
import nltk.corpus as corpus
from scipy.stats import kstest
from functools import reduce


# %matplotlib inline


# function for taking corpus key-word and returning the list-of-words and Title

def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]


def takecorp(corp):
    all_words_clean = read_docu(corp)
    if n_gram_value > 0:
        if n_gram_value == 1:
            name = 'Unigrams - '
        elif n_gram_value == 2:
            name = 'Digrams - '
        elif n_gram_value == 3:
            name = 'Trigrams - '
    elif n_gram_value == 0:
        name = ''
    if corp == igbb:
        Title = name + 'Igbo Bible'
    elif corp == YCB:
        Title = name + 'Yoruba Bible'
    elif corp == WEB:
        Title = name + 'World English Bible'
    elif corp == b:
        Title = name + 'Brown Corpus'
    elif corp == kjv:
        Title = name + 'King James Version'
    elif corp == bbcYr:
        Title = name + 'BBC Yoruba Corpus'
    elif corp == bbcigb:
        Title = name + 'BBC Igbo Corpus'
    elif corp == reut:
        Title = name + 'Reuters Corpus'

    return [all_words_clean, Title]


# In[1]:

# Best fit Zipf-Mandelbrot Hypothesis Trial Equation
def best_fit(x, m, b, c):
    return m * np.log(x + b) + c


def tokencount_aftercutoff(word_counts_n):
    return token_counts(word_counts_n)[1]


# Phython Function that reads the different files and filetypes giving out the full word-list
def read_docu(file):
    all_words = []

    if type(file) == nltk.text.Text:
        text = [str(file)]
        chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
        tr = str.maketrans("", "", chars_to_remove)
        text = [text[0].translate(tr)]
        if n_gram_value > 0:
            all_words = extract_ngrams(str(text), n_gram_value)
        elif n_gram_value == 0:
            for line in text:
                line = line.lower()
                line = line.strip().split()
                all_words += line
        return (all_words)

    elif type(file) == NLTKClass1 or type(file) == NLTKClass2 or type(file) == NLTKClass3:
        chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
        text = []
        for word in file.words():
            text.append(word)
        text = [str(text)]
        tr = str.maketrans("", "", chars_to_remove)
        text = [text[0].translate(tr)]
        if n_gram_value > 0:
            all_words = extract_ngrams(str(text), n_gram_value)
        elif n_gram_value == 0:
            for line in text:
                line = line.lower()
                line = line.strip().split()
                all_words += line
        return (all_words)

    else:
        with open(file, "r", encoding="utf-8") as input_file:
            text = [input_file.read()]
            chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
            tr = str.maketrans("", "", chars_to_remove)
            text = [text[0].translate(tr)]
            if n_gram_value > 0:
                all_words = extract_ngrams(str(text), n_gram_value)
            elif n_gram_value == 0:
                for line in text:
                    line = line.lower()
                    line = line.strip().split()
                    all_words += line
            return (all_words)


def corp_df_aftercut_off(all_words_clean, cutoff_freq):
    # Code to turn word-list into Data-Frame with rank, frequency and their logs

    word_list = all_words_clean
    words = pd.DataFrame(word_list, columns=['word'])
    word_counts1 = words.word.value_counts().reset_index()
    word_counts1.columns = ['word', 'n_raw']
    word_counts1['word_rank_raw'] = word_counts1.n_raw.rank(ascending=False)
    # i = word_counts['word_rank']
    # word_counts['log_n'] = [log(i,np.exp(1)) for i in word_counts.n]
    # word_counts['log_rank'] = [log(i,np.exp(1)) for i in word_counts.word_rank]

    if cutoff_freq > 0:
        n_raw = word_counts1.n_raw.tolist()
        # word_rank_raw = word_counts1.word_rank_raw.tolist()
        cut_off_index = n_raw.index(cutoff_freq)
        word_counts2 = word_counts1.iloc[:cut_off_index, :]
        word_counts2.columns = ['word2', 'n2', 'word_rank2']
        # words_trimmed = words.word[:cut_off_index].tolist()
    elif cutoff_freq == 0:
        word_counts2 = word_counts1
        word_counts2.columns = ['word2', 'n2', 'word_rank2']
        # word_counts2['n2'] = word_counts2.n_raw[:]
        # word_counts2['word_rank2'] = word_counts1.word_rank_raw[:]
        # words_trimmed = words.word.tolist()
    return word_counts2


def token_counts(word_counts_n):
    num_tokens = len(all_words_clean)
    num_types = len(set(all_words_clean))
    num_tokens_used = 0
    # print('word_counts[n] is ', word_counts2['n2'])
    for n in word_counts_n.tolist():
        num_tokens_used += n
    # print('num_tokens_used is ', num_tokens_used)
    # print('num of tokens from source corpus is ', num_tokens)
    print('No of tokens in ' + str(Title) + ' is ', num_tokens)
    print('No of tokens used after Freq. cutoff from the ' + str(Title) + ' for analysis is ', str(num_tokens_used))
    print('No of types in source corpus is ', num_types)
    return [num_tokens, num_tokens_used, num_types]


def bootstrap_dfs(all_words_clean, num_bootstraps, cutoff_freq):
    # Making Bootstraps
    word_counts2 = corp_df_aftercut_off(all_words_clean, cutoff_freq)
    df = word_counts2.iloc[:, :2]
    df.columns = ['bootword2', 'boot_n2']
    bootstraps_Dfs = [df]
    y_bootstraps = []
    boot_xs = []
    bootstraplists = []
    num_tokens = len(all_words_clean)
    for i in range(num_bootstraps):
        max_val = num_tokens
        size_of_corpus = num_tokens
        rand_word_indices = np.random.randint(0, max_val, size_of_corpus)  # an array of random indices
        arr_text = np.array(all_words_clean)  # need to use numpy as its waaaay quicker
        new_text = list(arr_text[rand_word_indices])  # quicker way to get the new corpus from the original corpus
        bootstrap_samp = new_text
        bootwords = pd.DataFrame(bootstrap_samp, columns=['bootword'])
        boot_counts = bootwords.bootword.value_counts().reset_index()
        boot_counts.columns = ['boot_word', 'boot_n_raw']
        boot_counts['boot_rank_raw'] = boot_counts.boot_n_raw.rank(ascending=False)

        if cutoff_freq > 0:
            boot_n_raw = boot_counts.boot_n_raw.tolist()
            # boot_rank_raw = boot_counts.boot_rank_raw.tolist()
            cut_off_index = boot_n_raw.index(cutoff_freq)
            # boot_counts2 = boots_trimmed.bootword2.value_counts().reset_index()
            boot_counts2 = boot_counts.iloc[:cut_off_index, :]
            boot_counts2.columns = ['bootword2', 'boot_n2', 'boot_rank2']
            # words_trimmed = words.word[:cut_off_index].tolist()
        elif cutoff_freq == 0:
            boot_counts2 = boot_counts
            boot_counts2['boot_n2'] = boot_counts.boot_n_raw[:]
            boot_counts2['boot_rank2'] = boot_counts.boot_rank_raw[:]
            # words_trimmed = words.word.tolist()
        bootstraps_Dfs.append(boot_counts2.iloc[:, :2])
        bootrank = boot_counts2.boot_rank2
        bootFreq = boot_counts2.boot_n2
        yvalues = np.log(bootFreq)
        xvalues = bootrank
        bootstraplists.append(bootstrap_samp)
        boot_xs.append(xvalues.tolist())
        y_bootstraps.append(yvalues.tolist())

    bootstrap_Dataframes = reduce(lambda left, right: pd.merge(left, right, on=['bootword2'], how="inner"),
                                  bootstraps_Dfs).fillna(0)
    return [bootstrap_Dataframes, boot_xs, y_bootstraps]


def y_uncert_hists(word_merged_Dataframe, n_most_freq_words, num_bins):
    for i in range(1, n_most_freq_words):
        word = word_merged_Dataframe.iloc[i, 0]
        values = word_merged_Dataframe.iloc[i, 1:].to_list()
        values = np.log(values)
        plt.hist(values, bins=num_bins)
        plt.title('rank = ' + str(i + 1) + '| word = ' + word)
        plt.show()


# In[2]:

# NLTK filetype Classes
NLTKClass1 = nltk.corpus.reader.tagged.CategorizedTaggedCorpusReader
NLTKClass2 = nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader
NLTKClass3 = nltk.corpus.util.LazyCorpusLoader

# Corpus Keys

# Complete bibles Yoruba YCB, Igbo igbb, English WEB & kjv
WEB = r'WEB.txt'
YCB = r'YCB.txt'
igbb = r'igbobible.txt'
kjv = r'KJV Gutenberg.txt'

# i = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Online Kaggle Zipf Testers\human_text.txt'
# j = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Online Kaggle Zipf Testers\robot_text.txt'
# YCB_OT = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\YCB OT.txt'
# YCB_NT = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\YCB NT.txt'
bbcYr = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Yoruba Bibeli Mimo\BBC Datasets\BBC Yoruba Articles\bbc yoruba raw1.txt'
# TED =  r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\TedTalks\TedTalks 2017.txt'
bbcigb = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\BBC Datasets\bbcigbo.txt'
b = brown
# wbtxt = webtext
# gut = gutenberg
reut = corpus.reuters
# inaug = corpus.inaugural


# Setting key Values
#
#
n_gram_value = 0  # ngram value, uni-gram n = 1, di-gram =2, tri-gram n = 3, n=0 uses written tokenizer as opposed to NLTK tokenizer (expected to yield identical results)
cutoff_freq = 5  # Cuts off words which occur at a frequency 10 and less
num_bootstraps = 10  # Gives the number of bootstraps which should be run
corp_key = igbb  # For picking the desired corpus using the keys above

# Calling takecorp() function, and defining all_words_clean word-list, to be called later
corp_words_and_Title = takecorp(corp_key)
all_words_clean = corp_words_and_Title[0]
Title = corp_words_and_Title[1]

# Calling Bootstrap DataFrame Function
bootstrap_Dfs = bootstrap_dfs(all_words_clean, num_bootstraps, cutoff_freq)
boot_Dfs = bootstrap_Dfs[0]
boot_xs = bootstrap_Dfs[1]
y_bootstraps = bootstrap_Dfs[2]

# print (bootstrap_Dfs)


'''
#For importing corpora combinations
all_words_clean1 = read_docu(wbtxt)
all_words_clean2 = read_docu(WEB)
all_words_clean3 = read_docu(b)
all_words_clean4 = read_docu(gut)

all_words_clean = all_words_clean1 + all_words_clean2 + all_words_clean3 + all_words_clean4

'''

word_counts2 = corp_df_aftercut_off(all_words_clean, cutoff_freq)

token_counts(word_counts2['n2'])

word_counts2['log_n'] = np.log(word_counts2.n2)
word_counts2['log_rank'] = np.log(word_counts2.word_rank2)

# Defining Plot Parameters - for trial function matplotlib.pyplot
log_n = word_counts2['log_n']
log_rank = word_counts2['log_rank']
# print(x)
xrank = word_counts2.word_rank2
# print ('y si ', log_n)
# print(word_counts['n']


# Plot Parameters - for final Plot


# Bootstrap Error Estimation
len_ybootstraps = []
for i in range(len(y_bootstraps)):
    len_ybootstraps.append(len(y_bootstraps[i]))

df_length = len(boot_Dfs['bootword2'])
# length = min(len_ybootstraps)
# length = int(len(y_bootstraps[0])*(4.5/5))


y_ith_errors = []
# i_val_lsts = []
print('len xrank is ', len(xrank))
print('len y_bootstraps[0] is ', len(y_bootstraps[0]))
print('len y_bootstraps[4] is ', len(y_bootstraps[4]))
# print('y_bootstraps[0][200] is ', y_bootstraps[0][200])
for k in range(1, df_length):
    i_vals = []
    yth_freq = boot_Dfs.iloc[k]
    for i in range(1, num_bootstraps + 1):
        yth_freq_i = np.log(yth_freq[i])
        i_vals.append(yth_freq_i)
    # var = statistics.variance(i_vals)
    # std = abs(var**.5)
    std = (np.std(i_vals))
    y_ith_errors.append(std)

# y_ith_errors = np.log(y_ith_errors).tolist()

print('length y_ith_errors is ', len(y_ith_errors))
# print('y_ith_errors list: ', y_ith_errors)
# print('length of 2 bootstrap is ', len(bootstraps[1]) + len(bootstraps[0]))


# For plotting using Plt
# f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(7, 7))
f, (ax1) = plt.subplots(1, figsize=(7, 7))
f.suptitle(Title)

ax1.set(xlim=(0, 12))
# sns.regplot("log_n", "log_rank", word_counts, ci=100, scatter_kws={"s": 50})


# Trimming xrank
x = log_rank
y = log_n
xrankraw = xrank

perct_err = [(x / y[y_ith_errors.index(x)]) for x in y_ith_errors]
# weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
weights = [1 / (x + 1e-50) for x in perct_err]
# weights = [1/(x+1e-19) for x in y_ith_errors]

length2 = min(len(x), len(weights))

print('len weights is ', len(weights))
print('len x is ', len(x))

x = log_rank[:length2]
y = np.log(boot_Dfs.iloc[1:, 1])
# y = log_n[:length2]
xrankraw = xrank
xrank = xrank[:length2]
weights = weights[:length2]

print('len new weights is ', len(weights))
print('len new x is ', len(x))
# perct_err = [(x/y[y_ith_errors.index(x)])for x in y_ith_errors]
# weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
# weights = [(1/(x+1e-9)) for x in perct_err]

m, c = np.polyfit(x, y, 1, w=weights)
ax1.set_xlabel('log(Rank)')
ax1.set_ylabel('log Frequency')
# print('gradient is ', m, 'and intercept is ', c)

print('bootstrap_length is ', num_bootstraps)
print('cut_off_frequency is ', cutoff_freq)

guess_parameters = [m, 12, c]
g = guess_parameters

n1 = len(xrank)
print('length of xrank is ', n1)
y1 = np.empty(n1)

# Trial_function of Zipf-Mandelbrot Guess Parameters for Eq of form  - m*(x+b) + c
for i in range(n1):
    m = g[0]
    b = g[1]
    c = g[2]
    y1[i] = best_fit(xrank[i], m, b, c)

# Best_fit Parameters for Trial Function of Zipf-Mandelbrot Equation
print('initial guess Parameters m, b, c : ', g)
# Curve_Fit Optimize
k, cov = curve_fit(best_fit, xrank, y, g)
print('curve_fit params m,b,c : ', k)

# Defining m = k[0] , b = k[1] & c = k[2] and Plot Curve_fitted Params
y2 = np.empty(n1)
for i in range(n1):
    y2[i] = best_fit(xrank[i], k[0], k[1], k[2])

m = k[0]
b = k[1]
c = k[2]

# Straight-line Pure Zipfian Estimate & mandelbrot corrected xlog for random test
straightline = m * x + c
xlog = np.log(xrank + k[1])
# yval = best_fit(x, m, b, c)

word_counts2['Zipf_log_y'] = [m * np.log(i + b) + c for i in xrankraw]
Zipf_log_rank = word_counts2['Zipf_log_y']

# Calculating m,b,c variance using Bootstrap (samples) estimation-technique
m_bootsamps = []
b_bootsamps = []
c_bootsamps = []
print('range 28 is ', range(28))

# Bootstrap Method
for i in range(num_bootstraps):
    xranksamp = boot_xs[i][:]
    ysamp = y_bootstraps[i][:]
    msamp, csamp = np.polyfit(np.log(xranksamp), ysamp, 1)
    guess = [msamp, 12, csamp]
    ksamp, cov_samp = curve_fit(best_fit, xranksamp, ysamp, guess)
    msamp = ksamp[0]
    bsamp = ksamp[1]
    csamp = ksamp[2]
    m_bootsamps.append(msamp)
    b_bootsamps.append(bsamp)
    c_bootsamps.append(csamp)

# print('m bootsamps are ', m_bootsamps)

m_boot_samples = m_bootsamps
mean_boot_m = (sum(m_boot_samples)) / len(m_boot_samples)
deviations_boot_m = [(x - m) ** 2 for x in m_boot_samples]
variance_boot_m = sum(deviations_boot_m) / len(m_boot_samples)
stdbootm = (variance_boot_m) ** .5
print('Mean Bootstraps m is ', mean_boot_m)
print('Var boot m is ', variance_boot_m)
print('std boot m is ', stdbootm)

b_boot_samples = b_bootsamps
mean_boot_b = (sum(b_boot_samples)) / len(b_boot_samples)
deviations_boot_b = [(x - b) ** 2 for x in b_boot_samples]
variance_boot_b = sum(deviations_boot_b) / len(b_boot_samples)
stdbootb = (variance_boot_b) ** .5
print('Mean Bootstraps b is ', mean_boot_b)
print('Var boot b is ', variance_boot_b)
print('std boot b is ', stdbootb)

c_boot_samples = c_bootsamps
mean_boot_c = (sum(c_boot_samples)) / len(c_boot_samples)
deviations_boot_c = [(x - c) ** 2 for x in c_boot_samples]
variance_boot_c = sum(deviations_boot_c) / len(c_boot_samples)
stdbootc = (variance_boot_c) ** .5
print('Mean Bootstraps c is ', mean_boot_c)
print('Var boot c is ', variance_boot_c)
print('std boot c is ', stdbootc)

error_m = abs(abs(mean_boot_m) - abs(m)) + abs(stdbootm)
error_b = abs(abs(mean_boot_b) - abs(b)) + abs(stdbootb)
error_c = abs(abs(mean_boot_c) - abs(c)) + abs(stdbootc)

# error_m = abs(stdbootm)
# error_b = abs(stdbootb)
# error_c = abs(stdbootc)


print('Error_m is ', error_m)
print('Error_b is ', error_b)
print('Error_c is ', error_c)

# Defining and plotting Maximum and minimum posisble Black-lines of Fit for Zipf-Mandelbrot Law using Estimated Variances

# ylower = (m + error_m)*np.log(xrank + (b+error_b)) + (c - error_c)
# yupper = (m - error_m)*np.log(xrank + abs(b-error_b)) + (c + error_c)
ylower = (m + error_m) * np.log(xrank + (b + error_b)) + (c - error_c)
yupper = (m - error_m) * np.log(xrank + abs(b - error_b)) + (c + error_c)
ytest = (m) * np.log(xrank + b) + (c)

ax1.plot(x, y, 'b+', markersize=2)
ax1.plot(x, y2, 'm')
ax1.plot(x, yupper, 'k')
ax1.plot(x, ylower, 'k')
# ax1.plot(x, ytest ,'k')
ax1.errorbar(x[:length2], y[:length2], yerr=y_ith_errors[:length2], fmt=' ')

y_uncert_hists(boot_Dfs, 5, 10)

'''

For plotting the m, b, c values estimated from all bootstraps


ax2.set_title('Bootstrap samples ' +' vs m_estimates')
ax2.plot(range(int(num_bootstraps)), m_bootsamps)
ax2.set_xlabel('bootstrap samples')
ax2.set_ylabel('m_boot_estimates')
#ax2.set(xlim=(0,4*n)) 
#ax2.set(ylim=(-1.78,-1.773)) 

ax3.set_title('Bootstrap samples ' +' vs b_estimates')
ax3.plot(range(int(num_bootstraps)), b_bootsamps)
ax3.set_xlabel('bootstrap samples')
ax3.set_ylabel('b_boot_estimates')
#ax3.set(xlim=(0,(4*n))) 
#ax3.set(ylim=(285,299))  


ax4.set_title('Bootstrap samples ' +' vs c_estimates')
ax4.plot(range(int(num_bootstraps)), c_bootsamps)
ax4.set_xlabel('bootstrap samples')
ax4.set_ylabel('c_boot_estimates') 
#ax4.set(xlim=(0,4*n)) 
#ax4.set(ylim=(17.90,18))  

'''

# Save Graphed Figure

plt.savefig(Title)

# print(y.rank(ascending=False))
print('kstest results are ', kstest(y[:length2], y2[:length2]))
# print('kstest results are ', kstest(y1[int(samplerange/n):int(samplerange)],straightline[int(samplerange/n):int(samplerange)]))
# print('kstest results are ', kstest(y_corrected[int(samplerange):int(samplerange*n)],straightline[int(samplerange/n):int(samplerange*n)]))


word_counts2['rel_dif'] = abs(Zipf_log_rank - y)

# relative_freqeuncy_log_rank_diff'
rel_diff = word_counts2['rel_dif']
word_counts2['perc_dif'] = 100 * (rel_diff / abs(Zipf_log_rank))
# print(word_counts2)

# Save Excel Table with Title
# word_counts2.to_excel(Title+'.xlsx')

print(stats.chisquare(f_obs=y[:length2], f_exp=y2[:length2]))
# print(stats.chisquare(f_obs = y1 , f_exp = Zipf_log_rank))


#Plot Parameters - for final Plot

   
    

#Bootstrap Error Estimation
len_ybootstraps = []
for i in range(len(y_bootstraps)):
    len_ybootstraps.append(len(y_bootstraps[i])) 


df_length = len(boot_Dfs['bootword2'])                       
#length = min(len_ybootstraps)
#length = int(len(y_bootstraps[0])*(4.5/5))


y_ith_errors = []
#i_val_lsts = []
print('len xrank is ', len(xrank))
print('len y_bootstraps[0] is ', len(y_bootstraps[0]))
print('len y_bootstraps[4] is ', len(y_bootstraps[4]))
#print('y_bootstraps[0][200] is ', y_bootstraps[0][200])
for k in range(1,df_length):   
    i_vals = []
    yth_freq = boot_Dfs.iloc[k]
    for i in range(1, num_bootstraps+1):
        yth_freq_i = np.log(yth_freq[i])
        i_vals.append(yth_freq_i)
    #var = statistics.variance(i_vals)
    #std = abs(var**.5)
    std = (np.std(i_vals))
    y_ith_errors.append(std)

#y_ith_errors = np.log(y_ith_errors).tolist()

print('length y_ith_errors is ', len(y_ith_errors))
#print('y_ith_errors list: ', y_ith_errors)  
#print('length of 2 bootstrap is ', len(bootstraps[1]) + len(bootstraps[0]))



# For plotting using Plt
#f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(7, 7))
f, (ax1) = plt.subplots(1, figsize=(7, 7))
f.suptitle(Title)


ax1.set(xlim=(0,12))
#sns.regplot("log_n", "log_rank", word_counts, ci=100, scatter_kws={"s": 50})


#Trimming xrank
x = log_rank
y = log_n
xrankraw = xrank


perct_err = [(x/y[y_ith_errors.index(x)])for x in y_ith_errors]
#weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
weights = [1/(x+1e-50) for x in perct_err]
#weights = [1/(x+1e-19) for x in y_ith_errors]

length2 = min(len(x),len(weights))

print('len weights is ', len(weights))
print('len x is ', len(x))

x = log_rank[:length2]
y = np.log(boot_Dfs.iloc[1:,1])
#y = log_n[:length2]
xrankraw = xrank
xrank = xrank[:length2]
weights = weights[:length2]


print('len new weights is ', len(weights))
print('len new x is ', len(x))
#perct_err = [(x/y[y_ith_errors.index(x)])for x in y_ith_errors]
#weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
#weights = [(1/(x+1e-9)) for x in perct_err]

m, c = np.polyfit(x, y, 1, w = weights)
ax1.set_xlabel('log(Rank)')
ax1.set_ylabel('log Frequency')
#print('gradient is ', m, 'and intercept is ', c) 

print('bootstrap_length is ', num_bootstraps)
print('cut_off_frequency is ', cutoff_freq)


guess_parameters = [m, 12, c]
g = guess_parameters


n1 = len(xrank)
print('length of xrank is ', n1)
y1 = np.empty(n1)


# Trial_function of Zipf-Mandelbrot Guess Parameters for Eq of form  - m*(x+b) + c
for i in range(n1):
    m = g[0]
    b = g[1]
    c = g[2]
    y1[i] = best_fit(xrank[i], m, b, c)


# Best_fit Parameters for Trial Function of Zipf-Mandelbrot Equation
print('initial guess Parameters m, b, c : ', g)
#Curve_Fit Optimize
k,cov = curve_fit(best_fit, xrank, y, g)
print('curve_fit params m,b,c : ', k)


# Defining m = k[0] , b = k[1] & c = k[2] and Plot Curve_fitted Params
y2 = np.empty(n1)
for i in range(n1):
    y2[i] = best_fit(xrank[i], k[0], k[1], k[2])
    
m = k[0]
b = k[1]
c = k[2]
    
#Straight-line Pure Zipfian Estimate & mandelbrot corrected xlog for random test
straightline = m*x + c
xlog = np.log(xrank + k[1])
#yval = best_fit(x, m, b, c)

word_counts2['Zipf_log_y'] = [m*np.log(i+b) + c for i in log_rank]
Zipf_log_rank = word_counts2['Zipf_log_y']



# Calculating m,b,c variance using Bootstrap (samples) estimation-technique
m_bootsamps = []
b_bootsamps = []
c_bootsamps = []
print('range 28 is ', range(28))


#Bootstrap Method 
for i in range(num_bootstraps):
    xranksamp = boot_xs[i][:]
    ysamp = y_bootstraps[i][:]
    msamp, csamp = np.polyfit(np.log(xranksamp), ysamp, 1)
    guess = [msamp, 12, csamp]
    ksamp,cov_samp = curve_fit(best_fit, xranksamp, ysamp, guess)
    msamp = ksamp[0]
    bsamp = ksamp[1]
    csamp = ksamp[2]
    m_bootsamps.append(msamp)
    b_bootsamps.append(bsamp)
    c_bootsamps.append(csamp)


#print('m bootsamps are ', m_bootsamps)

m_boot_samples = m_bootsamps
mean_boot_m = (sum(m_boot_samples))/len(m_boot_samples)
deviations_boot_m = [(x-m)**2 for x in m_boot_samples]
variance_boot_m = sum(deviations_boot_m)/len(m_boot_samples)
stdbootm = (variance_boot_m)**.5
print('Mean Bootstraps m is ', mean_boot_m)
print('Var boot m is ', variance_boot_m)
print('std boot m is ', stdbootm)

b_boot_samples = b_bootsamps
mean_boot_b = (sum(b_boot_samples))/len(b_boot_samples)
deviations_boot_b = [(x-b)**2 for x in b_boot_samples]
variance_boot_b = sum(deviations_boot_b)/len(b_boot_samples)
stdbootb = (variance_boot_b)**.5
print('Mean Bootstraps b is ', mean_boot_b)
print('Var boot b is ', variance_boot_b)
print('std boot b is ', stdbootb)

c_boot_samples = c_bootsamps
mean_boot_c = (sum(c_boot_samples))/len(c_boot_samples)
deviations_boot_c = [(x-c)**2 for x in c_boot_samples]
variance_boot_c = sum(deviations_boot_c)/len(c_boot_samples)
stdbootc = (variance_boot_c)**.5
print('Mean Bootstraps c is ', mean_boot_c)
print('Var boot c is ', variance_boot_c)
print('std boot c is ', stdbootc)


error_m = abs(abs(mean_boot_m)-abs(m)) + abs(stdbootm)
error_b = abs(abs(mean_boot_b)-abs(b)) + abs(stdbootb)
error_c = abs(abs(mean_boot_c)-abs(c)) + abs(stdbootc)

#error_m = abs(stdbootm)
#error_b = abs(stdbootb)
#error_c = abs(stdbootc)


print('Error_m is ', error_m)
print('Error_b is ', error_b)
print('Error_c is ', error_c)



# Defining and plotting Maximum and minimum posisble Black-lines of Fit for Zipf-Mandelbrot Law using Estimated Variances

#ylower = (m + error_m)*np.log(xrank + (b+error_b)) + (c - error_c)
#yupper = (m - error_m)*np.log(xrank + abs(b-error_b)) + (c + error_c)
ylower = (m + error_m)*np.log(xrank + (b+error_b)) + (c - error_c)
yupper = (m - error_m)*np.log(xrank + abs(b - error_b)) + (c + error_c)
ytest = (m)*np.log(xrank+b) + (c)

ax1.plot(x, y, 'b+', markersize = 2)
ax1.plot(x, y2, 'm')
ax1.plot(x, yupper ,'k')
ax1.plot(x, ylower ,'k')
#ax1.plot(x, ytest ,'k')
ax1.errorbar(x[:length2], y[:length2], yerr = y_ith_errors[:length2], fmt = ' ')


    
y_uncert_hists(boot_Dfs, 5, 10)


'''

For plotting the m, b, c values estimated from all bootstraps


ax2.set_title('Bootstrap samples ' +' vs m_estimates')
ax2.plot(range(int(num_bootstraps)), m_bootsamps)
ax2.set_xlabel('bootstrap samples')
ax2.set_ylabel('m_boot_estimates')
#ax2.set(xlim=(0,4*n)) 
#ax2.set(ylim=(-1.78,-1.773)) 

ax3.set_title('Bootstrap samples ' +' vs b_estimates')
ax3.plot(range(int(num_bootstraps)), b_bootsamps)
ax3.set_xlabel('bootstrap samples')
ax3.set_ylabel('b_boot_estimates')
#ax3.set(xlim=(0,(4*n))) 
#ax3.set(ylim=(285,299))  


ax4.set_title('Bootstrap samples ' +' vs c_estimates')
ax4.plot(range(int(num_bootstraps)), c_bootsamps)
ax4.set_xlabel('bootstrap samples')
ax4.set_ylabel('c_boot_estimates') 
#ax4.set(xlim=(0,4*n)) 
#ax4.set(ylim=(17.90,18))  

'''


# Save Graphed Figure

plt.savefig(Title)


#print(y.rank(ascending=False))
#print('kstest results are ', kstest(y[:length2],y2[:length2]))
print('kstest results are ', kstest(y[:200],y2[:200]))
#print('kstest results are ', kstest(y1[int(samplerange/n):int(samplerange)],straightline[int(samplerange/n):int(samplerange)]))
#print('kstest results are ', kstest(y_corrected[int(samplerange):int(samplerange*n)],straightline[int(samplerange/n):int(samplerange*n)]))
  

word_counts2['rel_dif'] = abs(Zipf_log_rank - y)

#relative_freqeuncy_log_rank_diff'
rel_diff = word_counts2['rel_dif']
word_counts2['perc_dif'] = 100 * (rel_diff/abs(Zipf_log_rank))
#print(word_counts2)

#Save Excel Table with Title
#word_counts2.to_excel(Title+'.xlsx')

print(stats.chisquare(f_obs = y[:length2] , f_exp = y2[:length2]))
#print(stats.chisquare(f_obs = y1 , f_exp = Zipf_log_rank))

