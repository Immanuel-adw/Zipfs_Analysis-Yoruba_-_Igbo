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
import nltk.corpus as corpus
from scipy.stats import kstest
#%matplotlib inline





# function for taking corpus key-word and returning the list-of-words and Title

def takecorp(corp):
    all_words_clean = read_docu(corp)
    if corp == igbb:
        Title = 'Igbo Bible'
    elif corp == YCB:
        Title = 'Yoruba Bible'
    elif corp == WEB:
        Title = 'World English Bible'
    #elif corp == b:
        #Title = 'Brown Corpus'
    elif corp == kjv:
        Title = 'King James Version'
    #elif corp == bbcYr:
        #Title = 'BBC Yoruba Corpus'
    #elif corp == reut:
        #Title = 'Reuters Corpus'
    
    return [all_words_clean, Title]


#Best fit Zipf-Mandelbrot Hypothesis Trial Equation 
def best_fit(x, m, b, c):
    return m*np.log(x+b) + c   


#Phython Function that reads the different files and filetypes giving out the full word-list
def read_docu(file):
    all_words = []
    
    if type(file) == nltk.text.Text:
        text = [str(file)]
        chars_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"
        tr = str.maketrans("", "", chars_to_remove) 
        text = [text[0].translate(tr)]
        #text = str(text)
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
        #print(text)
        #text = str(text)
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
            #text = str(text)
            for line in text:
                line = line.lower()
                line = line.strip().split()
                all_words += line
            return(all_words)               


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
#bbcYr = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Yoruba Bibeli Mimo\BBC Datasets\BBC Yoruba Articles\bbc yoruba raw1.txt'
#TED =  r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\TedTalks\TedTalks 2017.txt'
#bbcigb = r'C:\Users\imman\Documents\ICL\Year 3\Term 1\Term 1 Project\Python\Other Corpora\BBC Datasets\bbcigbo.txt'
#b = brown
#wbtxt = webtext
#gut = gutenberg
#reut = c.reuters
#inaug = c.inaugural 


#NLTK filetype Classes
NLTKClass1 = nltk.corpus.reader.tagged.CategorizedTaggedCorpusReader
NLTKClass2 = nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader
NLTKClass3 = nltk.corpus.util.LazyCorpusLoader



# Calling takecorp() function, and defining all_words_clean word-list, to be called later
corp_words_and_Title = takecorp(kjv)
all_words_clean = corp_words_and_Title[0]
Title = corp_words_and_Title[1]


'''
#For importing corpora combinations
all_words_clean1 = read_docu(wbtxt)
all_words_clean2 = read_docu(WEB)
all_words_clean3 = read_docu(b)
all_words_clean4 = read_docu(gut)

all_words_clean = all_words_clean1 + all_words_clean2 + all_words_clean3 + all_words_clean4

'''


#Code to turn word-list into Data-Frame with rank, frequency and their logs
cutoff_freq = 5

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
    word_rank_raw = word_counts1.word_rank_raw.tolist()
    cut_off_index = n_raw.index(cutoff_freq)
    new_bag_of_words = []
    for i in range(cut_off_index):
        for k in range(word_counts1.n_raw[i]):
            new_bag_of_words.append(word_counts1.word[i])
    words_trimmed = pd.DataFrame(new_bag_of_words, columns = ['word2'])
    word_counts2 = words_trimmed.word2.value_counts().reset_index()
    word_counts2.columns = ['word2', 'n2']
    word_counts2['word_rank2'] = word_counts2.n2.rank(ascending=False)
    #words_trimmed = words.word[:cut_off_index].tolist()
elif cutoff_freq == 0:
    word_counts2['n2'] = word_counts1.n_raw[:]
    word_counts2['word_rank2'] = word_counts1.word_rank_raw[:]
    #words_trimmed = words.word.tolist()

word_counts2['log_n'] = np.log(word_counts2.n2)
word_counts2['log_rank'] = np.log(word_counts2.word_rank2)



# Defining Plot Parameters - for trial function matplotlib.pyplot 
log_n = word_counts2['log_n']
log_rank = word_counts2['log_rank']
#print(x)
xrank = word_counts2.word_rank2
#print ('y si ', log_n)
#print(word_counts['n']

#Plot Parameters - for final Plot

y_err = []
y_bootstraps = []
boot_xs = []
bootstraplists = []
#num_tokens = sum(word_counts.n.tolist())®®
num_tokens = len(all_words_clean)
num_tokens_used = 0
print('word_counts[n] is ', word_counts2['n2'])
for n in word_counts2['n2'].tolist():
    num_tokens_used+=n
print('num_tokens_used is ', num_tokens_used)   

#Making Bootstraps
bootstraplength = 10
print('No of tokens in '+ str(Title)+' is ', len(all_words_clean))
print('No of tokens taken from the '+ str(Title)+' for analysis is ', str(num_tokens_used))
print('No of types is ', len(set(all_words_clean)))
for i in range(bootstraplength):

    bootstrapsamp = []
    for i in range(num_tokens):
        nrand = np.random.randint(num_tokens)
        bootstrapsamp.append(all_words_clean[nrand])
        
    bootwords = pd.DataFrame(bootstrapsamp, columns=['bootword'])
    boot_counts = bootwords.bootword.value_counts().reset_index()
    boot_counts.columns = ['boot word', 'boot_n_raw']
    boot_counts['boot_rank_raw'] = boot_counts.boot_n_raw.rank(ascending=False)
    
    if cutoff_freq > 0:
        boot_n_raw = boot_counts.boot_n_raw.tolist()
        cut_off_index = boot_n_raw.index(cutoff_freq)
        boot_counts['boot_n'] = boot_counts.boot_n_raw[:cut_off_index]
        boot_counts['boot_rank'] = boot_counts.boot_rank_raw[:cut_off_index]
        bootwords_trimmed = bootwords.bootword[:cut_off_index].tolist()
    elif cutoff_freq == 0:
        boot_counts['boot_n'] = boot_counts.n_raw[:]
        boot_counts['boot_rank'] = boot_counts.word_rank_raw[:]
        bootwords_trimmed = bootwords.bootword.tolist()
        
    bootrank = boot_counts.boot_rank
    yvalues = np.log(boot_counts.boot_n)
    bootstraplists.append(bootstrapsamp)
    boot_xs.append(bootrank.tolist())
    y_bootstraps.append(yvalues.tolist())




#Bootstrap Error Estimation
len_ybootstraps = []
for i in range(len(y_bootstraps)):
    len_ybootstraps.append(len(y_bootstraps[i])) 
                       
length = min(len_ybootstraps)
#length = int(len(y_bootstraps[0])*(4.5/5))
y_ith_errors = []
#i_val_lsts = []
print('len xrank is ', len(xrank))
print('len y_bootstraps[0] is ', len(y_bootstraps[0]))
print('len y_bootstraps[4] is ', len(y_bootstraps[4]))
#print('y_bootstraps[0][200] is ', y_bootstraps[0][200])
for k in range(length):   
    i_vals = []
    for i in range(bootstraplength):
        i_vals.append(y_bootstraps[i][k])
    var = statistics.variance(i_vals)
    std = abs(var**.5)
    y_ith_errors.append(std)
print('length ith_errors is ', len(y_ith_errors))
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
xrank = xrank

x = log_rank[:length]
y = log_n[:length]
xrankraw = xrank
xrank = xrank[:length]

weights = [1/(x+1e-9) for x in y_ith_errors]
weights = weights[:length]
print('len weights is ', len(weights))
print('len x is ', len(x))
#perct_err = [(x/y[y_ith_errors.index(x)])for x in y_ith_errors]
#weights = [1 - ((x/y[y_ith_errors.index(x)])/sum(perct_err)) for x in y_ith_errors]
#weights = [(1/(x+1e-9)) for x in perct_err]

m, c = np.polyfit(x, y, 1, w = weights)
ax1.set_xlabel('log(Rank)')
ax1.set_ylabel('logN')
#print('gradient is ', m, 'and intercept is ', c) 

print('bootstrap_length is ', bootstraplength)
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

word_counts2['Zipf_log_y'] = [m*np.log(i+b) + c for i in xrankraw]
Zipf_log_rank = word_counts2['Zipf_log_y']



# Calculating m,b,c variance using Bootstrap (samples) estimation-technique
m_bootsamps = []
b_bootsamps = []
c_bootsamps = []
print('range 28 is ', range(28))


#Bootstrap Method 
for i in range(bootstraplength):
    xranksamp = boot_xs[i][:length]
    ysamp = y_bootstraps[i][:length]
    msamp, csamp = np.polyfit(np.log(xranksamp), ysamp, 1)
    guess = [msamp, 12, csamp]
    ksamp,cov_samp = curve_fit(best_fit, xranksamp, ysamp, g)
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
ax1.errorbar(x[:length], y[:length], yerr = y_ith_errors, fmt = ' ')



'''

For plotting the m, b, c values estimated from all bootstraps


ax2.set_title('Bootstrap samples ' +' vs m_estimates')
ax2.plot(range(int(bootstraplength)), m_bootsamps)
ax2.set_xlabel('bootstrap samples')
ax2.set_ylabel('m_boot_estimates')
#ax2.set(xlim=(0,4*n)) 
#ax2.set(ylim=(-1.78,-1.773)) 

ax3.set_title('Bootstrap samples ' +' vs b_estimates')
ax3.plot(range(int(bootstraplength)), b_bootsamps)
ax3.set_xlabel('bootstrap samples')
ax3.set_ylabel('b_boot_estimates')
#ax3.set(xlim=(0,(4*n))) 
#ax3.set(ylim=(285,299))  


ax4.set_title('Bootstrap samples ' +' vs c_estimates')
ax4.plot(range(int(bootstraplength)), c_bootsamps)
ax4.set_xlabel('bootstrap samples')
ax4.set_ylabel('c_boot_estimates') 
#ax4.set(xlim=(0,4*n)) 
#ax4.set(ylim=(17.90,18))  

'''


# Save Graphed Figure

plt.savefig(Title)


#print(y.rank(ascending=False))


word_counts2['rel_dif'] = abs(Zipf_log_rank - y)

#relative_freqeuncy_log_rank_diff'
rel_diff = word_counts2['rel_dif']
word_counts2['perc_dif'] = 100 * (rel_diff/abs(Zipf_log_rank))
#print(word_counts)

#Save Excel Table with Title
#word_counts.to_excel(Title+'.xlsx')

print(stats.chisquare(f_obs = y , f_exp = y1))
#print(stats.chisquare(f_obs = y1 , f_exp = Zipf_log_rank))

