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
import statsmodels.api as sm
from statsmodels.formula.api import ols
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
    num_tokens_used = sum(word_counts_n.tolist())
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
            if cutoff_freq in boot_n_raw:
                cut_off_index = boot_n_raw.index(cutoff_freq)
            else:
                cutoff_freq = cutoff_freq - 1
                if cutoff_freq in boot_n_raw:
                    cut_off_index = boot_n_raw.index(cutoff_freq)
                else:
                    cutoff_freq = cutoff_freq - 1
                    cut_off_index = boot_n_raw.index(cutoff_freq)
            #boot_counts2 = boots_trimmed.bootword2.value_counts().reset_index()
            boot_counts2 = boot_counts.iloc[:cut_off_index,:]
            boot_counts2.columns = ['bootword2', 'boot_n2', 'boot_rank2']
            #words_trimmed = words.word[:cut_off_index].tolist()
        elif cutoff_freq == 0:
            boot_counts2 = boot_counts
            boot_counts2.columns = ['bootword2', 'boot_n2', 'boot_rank2']
            
            #words_trimmed = words.word.tolist()
        bootstraps_Dfs.append(boot_counts2.iloc[:,:2]) 
        bootrank = boot_counts2.boot_rank2
        bootFreq = boot_counts2.boot_n2
        yvalues = bootFreq
        xvalues = bootrank
        bootstraplists.append(bootstrap_samp)
        boot_xs.append(xvalues.tolist())
        y_bootstraps.append(yvalues.tolist())
        
    bootstrap_Dataframes = reduce(lambda  left,right: pd.merge(left,right,on=['bootword2'], how="inner"), bootstraps_Dfs).fillna(0)
    return [bootstrap_Dataframes, boot_xs, y_bootstraps]


def y_uncert_hists(word_merged_Dataframe, n_most_freq_words, num_bins):
    for i in range(n_most_freq_words):
        fig, ax3  = plt.subplots(1, figsize=(6.5, 6.5))
        fig.suptitle(Title + ' Bootstrap Histograms')
        word = word_merged_Dataframe.iloc[i,0]
        values = word_merged_Dataframe.iloc[i,1:].to_list()
        values = np.log(values) 
        ax3.hist(values, bins = num_bins)
        hist_title = ' rank '+ str(i + 1) +' word ' + str(word)
        plt.title( str(hist_title) )
        plt.savefig(Title + hist_title + ID + ' Bootstrap Hist' )
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
cutoff_freq =  70    # Cuts off words which occur at a frequency 10 and less
num_bootstraps = 1000  # Gives the number of bootstraps which should be run
corp_key = YCB  # For picking the desired corpus using the keys above

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

# In[]:
    
num_tokens = len(all_words_clean)
num_types = len(set(all_words_clean))
num_tokens_used = sum(word_counts2['n2'].tolist())
#print('word_counts[n] is ', word_counts2['n2'])



# In[]:

# Plot Parameters - for final Plot


# Bootstrap Error Estimation
'''
len_ybootstraps = []
for i in range(len(y_bootstraps)):
    len_ybootstraps.append(len(y_bootstraps[i]))
'''
df_length = len(boot_Dfs['bootword2'])
# length = min(len_ybootstraps)
# length = int(len(y_bootstraps[0])*(4.5/5))


y_ith_errors = []
# i_val_lsts = []
#print('len xrank is ', len(xrank))
#print('len y_bootstraps[0] is ', len(y_bootstraps[0]))
#print('len y_bootstraps[4] is ', len(y_bootstraps[4]))
# print('y_bootstraps[0][200] is ', y_bootstraps[0][200])
for k in range(1, df_length):
    i_vals = []
    yth_freq = boot_Dfs.iloc[k, 1: num_bootstraps + 1].tolist()
    i_vals.append(yth_freq)
    i_vals = np.log(i_vals)
    # var = statistics.variance(i_vals)
    # std = abs(var**.5)
    std = (np.std(i_vals))
    y_ith_errors.append(std)

# y_ith_errors = np.log(y_ith_errors).tolist()


# print('y_ith_errors list: ', y_ith_errors)
# print('length of 2 bootstrap is ', len(bootstraps[1]) + len(bootstraps[0]))


# For plotting using Plt
# f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(7, 7))
f, (ax1, ax2) = plt.subplots(2, figsize=(7, 14))
f.suptitle(Title)

ax1.set(xlim=(0, 12))
# sns.regplot("log_n", "log_rank", word_counts, ci=100, scatter_kws={"s": 50})


# Trimming xrank
x = log_rank
y = log_n
xrankraw = xrank

perct_err = [(x / y[y_ith_errors.index(x)]) for x in y_ith_errors]
weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
#weights = [1 / (x + 1e-50) for x in perct_err]
#weights = [1/(x+1e-19) for x in y_ith_errors]

length2 = min(len(x), len(weights))

#print('len weights is ', len(weights))
#print('len x is ', len(x))



x = log_rank[:length2]
len_y = len(boot_Dfs.iloc[1:, 1])
x_raw = np.arange(1,len(boot_Dfs.iloc[1:, 1])+1)
x = np.log(x_raw)
y = np.log(boot_Dfs.iloc[1:, 1])
# y = log_n[:length2]
xrankraw = xrank
xrank = xrank[:length2]
weights = weights[:]


# perct_err = [(x/y[y_ith_errors.index(x)])for x in y_ith_errors]
# weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
# weights = [(1/(x+1e-9)) for x in perct_err]

m1, c1 = np.polyfit(x, y, 1, w=weights)
# print('gradient is ', m, 'and intercept is ', c)



guess_parameters = [m1, 12, c1]
g = guess_parameters

n1 = len_y
#print('length of xrank is len x is ', n1)
y1 = np.empty(n1)

# Trial_function of Zipf-Mandelbrot Guess Parameters for Eq of form  - m*(x+b) + c
for i in range(n1):
    m = g[0]
    b = g[1]
    c = g[2]
    y1[i] = best_fit(x_raw[i], m, b, c)

# Best_fit Parameters for Trial Function of Zipf-Mandelbrot Equation
print('initial guess Parameters m, b, c : ', g)



# Curve_Fit Optimize
k, cov = curve_fit(best_fit, x_raw, y, g)
#print('curve_fit params m,b,c : ', k)



# Defining m = k[0] , b = k[1] & c = k[2] and Plot Curve_fitted Params
y2 = np.empty(n1)
for i in range(n1):
    y2[i] = best_fit(x_raw[i], k[0], k[1], k[2])

m = k[0]
b = k[1]
c = k[2]

# Straight-line Pure Zipfian Estimate & mandelbrot corrected xlog for random test
#straightline = m1 * x + c1
straightline = m1 * x + c1
#xlog = np.log(xrank + k[1])
# yval = best_fit(x, m, b, c)

word_counts2['Zipf_log_y'] = [m * np.log(i + b) + c for i in xrankraw]
Zipf_log_rank = word_counts2['Zipf_log_y']

# Calculating m,b,c variance using Bootstrap (samples) estimation-technique
m_bootsamps = []
b_bootsamps = []
c_bootsamps = []
m_bootsamps_straight = []
c_bootsamps_straight = []

# Bootstrap Method
for i in range(num_bootstraps):
    print('remaining = ' + str(num_bootstraps - i))
    xranksamp = boot_xs[i][:]
    ysamp = y_bootstraps[i][:]
    msamp1, csamp1 = np.polyfit(np.log(xranksamp), np.log(ysamp), 1)
    guess = [msamp1, 12, csamp1]
    ksamp, cov_samp = curve_fit(best_fit, xranksamp, ysamp, guess)
    msamp = ksamp[0]
    bsamp = ksamp[1]
    csamp = ksamp[2]
    m_bootsamps.append(msamp)
    b_bootsamps.append(bsamp)
    c_bootsamps.append(csamp)
    m_bootsamps_straight.append(msamp1)
    c_bootsamps_straight.append(csamp1)

# print('m bootsamps are ', m_bootsamps)

print('curve_fit params m,b,c : ', k)
print('length y_ith_errors is ', len(y_ith_errors))
print('len new weights is ', len(weights))
print('len new x is ', len(x))
print('bootstrap_length is ', num_bootstraps)
print('cut_off_frequency is ', cutoff_freq)


m_boot_samples = m_bootsamps_straight
mean_boot_m = (sum(m_boot_samples)) / len(m_boot_samples)
deviations_boot_m = [(x - m) ** 2 for x in m_boot_samples]
variance_boot_m = sum(deviations_boot_m) / len(m_boot_samples)
stdbootm = (variance_boot_m) ** .5
print('Mean Bootstraps m_straight is ', mean_boot_m)
print('Var boot m_straight is ', variance_boot_m)
print('std boot m_straight is ', stdbootm)

c_boot_samples = c_bootsamps_straight
mean_boot_c = (sum(c_boot_samples)) / len(c_boot_samples)
deviations_boot_c = [(x - c) ** 2 for x in c_boot_samples]
variance_boot_c = sum(deviations_boot_c) / len(c_boot_samples)
stdbootc = (variance_boot_c) ** .5
print('Mean Bootstraps c_straight is ', mean_boot_c)
print('Var boot c_straight is ', variance_boot_c)
print('std boot c_straight is ', stdbootc)
error_m_straight = + abs(stdbootm)
error_c_straight = + abs(stdbootc)


ID = ' Cutoff Freq ' + str(cutoff_freq) + ' ' + str(num_bootstraps) + ' Bootstraps'

kstest_results_straight = kstest(y[:length2], straightline[:length2])

print('kstest results Zipf-General are ', kstest_results_straight)

array_straight = [[Title + ' Zipf-General fit', 'Original Value', 'Boot Mean', 'Variance', 'Standard Deviation'], ['Gradient', m1, mean_boot_m, variance_boot_m, stdbootm ], ['Intercept', c1, mean_boot_c, variance_boot_c, stdbootc], ['Num of Tokens', num_tokens], ['Num of Tokens Used', num_tokens_used],['Num of Types', num_types ], ['Cut-off Freq', cutoff_freq], ['length of Weights and Data Points Measured', len(weights)], ['Number of Bootstraps', num_bootstraps], ['kstest results Zipf-General fit', str(kstest_results_straight)]]

params = pd.DataFrame(array_straight).T
params.to_excel(Title + ' Zipf-General Parameters ' + ID + '.xlsx')


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

#error_m = abs(abs(mean_boot_m) - abs(m)) + abs(stdbootm)
#error_b = abs(abs(mean_boot_b) - abs(b)) + abs(stdbootb)
#error_c = abs(abs(mean_boot_c) - abs(c)) + abs(stdbootc)

error_m = abs(stdbootm)
error_b = abs(stdbootb)
error_c = abs(stdbootc)


print('Error_m is ', error_m)
print('Error_b is ', error_b)
print('Error_c is ', error_c)



kstest_results_Mandelbrot = kstest(y[:length2], y2[:length2])

print('kstest results Zipf-Mandelbrot are ', kstest_results_Mandelbrot)

array_Mandelbrot = [[Title + ' Zipf-Mandelbrot fit', 'Original Value', 'Boot Mean', 'Variance', 'Standard Deviation'], ['Gradient', m, mean_boot_m, variance_boot_m, stdbootm ], ['Intercept', c1, mean_boot_c, variance_boot_c, stdbootc], ['Num of Tokens', num_tokens], ['Num of Tokens Used', num_tokens_used],['Num of Types', num_types ], ['Cut-off Freq', cutoff_freq], ['length of Weights and Data Points Measured', len(weights)], ['Number of Bootstraps', num_bootstraps], ['kstest results Zipf-Mandelbrot fit', str(kstest_results_Mandelbrot)]]

params = pd.DataFrame(array_Mandelbrot).T
params.to_excel(Title + ' Zipf-Mandelbrot Parameters ' + ID + '.xlsx')

# Defining and plotting Maximum and minimum posisble Black-lines of Fit for Zipf-Mandelbrot Law using Estimated Variances

# ylower = (m + error_m)*np.log(xrank + (b+error_b)) + (c - error_c)
# yupper = (m - error_m)*np.log(xrank + abs(b-error_b)) + (c + error_c)
ylower = (m + error_m) * np.log(xrank + (b + error_b)) + (c - error_c)
yupper = (m - error_m) * np.log(xrank + abs(b - error_b)) + (c + error_c)
ytest = (m) * np.log(xrank + b) + (c)
max_grad = m1 - error_m_straight
min_grad = m1 + error_m_straight
ylower1 = (m1 + error_m_straight) * x + (c1)
yupper1 = (m1 - error_m_straight) * x + (c1)
#ytest = (m) * np.log(xrank + b) + (c)


ax1.plot(x, y, 'b+', markersize=3, label = 'data-points')
ax1.plot(x, straightline, 'm', label = 'Zipf-fit: Gradient = '+ str(round(m1,1)) + ' +/- ' + str(round(error_m_straight, 1)))
ax2.plot([], [], 'm', label = 'log(Freq) = '+ str(round(m1,1)) + ' x log(Rank) '  + '+ ' + str(round(c1,1)))
ax1.plot(x, yupper1, 'k', label = 'Max. Gradient = ' + str(round(max_grad,1)))
ax1.plot(x, ylower1, 'k', label = 'Min. Gradient = ' + str(round(min_grad,1))) #+'\n '+str(kstest_results_straight))
ax1.errorbar(x[:length2], y[:length2], yerr=y_ith_errors[:length2], fmt=' ')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('log(Rank)')
ax1.set_ylabel('log Frequency')

#Mandelbrot Plot
y_mandel = y2
ax2.plot(x, y, 'b+', markersize=2.8, label = 'data-points')
ax2.plot(x, y_mandel, 'm', label = 'Zipf-Mandelbrot fit: Gradient = '+ str(round(m,1)) + ' +/- ' + str(round(error_m_straight, 1)))
ax2.plot([], [], 'm', label = 'log(Freq) = '+ str(round(m,1)) + ' x log(Rank + ' + str(round(b,1)) + ') ' + '+ ' + str(round(c,1)))
ax2.errorbar(x[:length2], y[:length2], yerr=y_ith_errors[:length2], fmt=' ')
ax2.legend(loc = 'upper right')
ax2.set_xlabel('log(Rank)')
ax2.set_ylabel('log(Frequency)')


# Save Graphed Figure
plt.savefig(Title + ID)

# In[]:
# Residuals Plots
arr = {'log_Freq' : y, 'log_Rank' : x}
regress = pd.DataFrame(arr)

model = ols('log_Freq~log_Rank', data = regress).fit()
fig2 = plt.figure(figsize=(18,12))
fig2 = sm.graphics.plot_regress_exog(model, 'log_Rank', fig=fig2)

plt.savefig(Title + ' Residuals Plots' + ID)

# In[]: #Uncertainty Histogram Plots

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


# print(y.rank(ascending=False))
# print('kstest results are ', kstest(y1[int(samplerange/n):int(samplerange)],straightline[int(samplerange/n):int(samplerange)]))
# print('kstest results are ', kstest(y_corrected[int(samplerange):int(samplerange*n)],straightline[int(samplerange/n):int(samplerange*n)]))


word_counts2['rel_dif'] = abs(Zipf_log_rank - y)

# relative_freqeuncy_log_rank_diff'
rel_diff = word_counts2['rel_dif']
word_counts2['perc_dif'] = 100 * (rel_diff / abs(Zipf_log_rank))
# print(word_counts2)

# Save Excel Table with Title
word_counts2.to_excel(Title + ID +'.xlsx')
boot_Dfs.to_excel(Title + ' Bootstraps Table' + ID + '.xlsx')
#print(stats.chisquare(f_obs=y[:length2], f_exp=y2[:length2]))
# print(stats.chisquare(f_obs = y1 , f_exp = Zipf_log_rank))


#Plot Parameters - for final Plot