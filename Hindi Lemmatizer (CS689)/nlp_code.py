#!/usr/bin/env python
# coding: utf-8

# In[1]:

enter_word = input("शब्द दर्ज करें : ")
# add default dict
from collections import defaultdict
di = defaultdict()


# In[2]:


# reading file and saving words in dictionary
file1 = open("trainLemmaToWordMap.csv",encoding="utf8")
count = 1
for i in file1:
    if 'I' in i[0]:
        continue
    r = i.split(",")
    root = r[1]
    word = r[2].replace('\n','')
    if word in di.keys():
        continue
    else:
        di[word] = root


# In[3]:


# reading file and saving words in dictionary
file2 = open("testLemmaToWordMap.csv",encoding="utf8")
count = 1
for i in file2:
    if 'I' in i[0]:
        continue
    r = i.split(",")
    root = r[1]
    word = r[2].replace('\n','')
    if word in di.keys():
        continue
    else:
        di[word] = root


# In[4]:


# reading file and saving words in dictionary
file3 = open("devLemmaToWordMap.csv",encoding="utf8")
count = 1
for i in file3:
    if 'I' in i[0]:
        continue
    r = i.split(",")
    root = r[1]
    word = r[2].replace('\n','')
    if word in di.keys():
        continue
    else:
        di[word] = root


# In[5]:


# create syllables of a word
def syllables(word):
        signs = [u'\u0902', u'\u093e', u'\u093f', u'\u0940', 
                 u'\u0941', u'\u0942', u'\u0943', u'\u0944', u'\u0946',
                 u'\u0947', u'\u0948', u'\u0949', u'\u094A', u'\u094B', u'\u094C',
                 u'\u094d']
        limiters = ['.', '`', '!', ';', ', ', '?']

        virama = signs[-1] 
        lst_chars = []
        for char in word:
            if char in limiters:
                lst_chars.append(char)
            elif char in signs:
                if lst_chars:
                    lst_chars[-1] = lst_chars[-1] + char
                else:
                    continue
            else:
                if len(lst_chars) == 0:
                    lst_chars.append(char)
                else:
                    if lst_chars[-1][-1] == virama:
                        lst_chars[-1] = lst_chars[-1] + char
                    else:
                        lst_chars.append(char)
        return lst_chars


# In[6]:


# function to return root word from input word
def root_word(word):
    
    # check input word is given in the dict
    if word in di.keys():
        return di[word]
    
    # list of common suffixes
    suffix_list = {
    1: ["ो","े","ू","ु","ि","ऊ"],
    2: ["कर","ाओ","िए","ाई","ाए","ने","नी","ना","ते","ीं","ती","ता","ाँ","ां","ों","ें","ीय","गर","ाऊ"],
    3: ["ाकर","ाइए","ाईं","ाया","ेगी","ेगा","ोगी","ोगे","ाने","ाना","ाते","ाती","ाता","तीं","ाओं","ाएं","ुओं","ुएं","ुआं","नीय"
       ,"पूर्ण","पन","िया","दार"],
    4: ["ाएगी","ाएगा","ाओगी","ाओगे","एंगी","ेंगी","एंगे","ेंगे","ूंगी","ूंगा","ातीं","नाओं","नाएं","ताओं","ताएं","ियाँ","ियों","ियां"],
    5: ["ाएंगी","ाएंगे","ाऊंगी","ाऊंगा","ाइयाँ","ाइयों","ाइयां","ीयता","ियत"],
    6: ["शाली","वाला","क्कड़"]
        }
    
    # list of rules to derive root word with the help of suffix stripping / appending
    rule_based_list = [suffix_list[4][17], suffix_list[4][16], suffix_list[1][1], 'ा', 'ी', suffix_list[5][4], 
                      suffix_list[5][5], suffix_list[5][6]]

    # rule 1
    if word.endswith(rule_based_list[0]):
        return word[:word.rindex(rule_based_list[0])] + 'ी'
    
    # rule 2
    elif word.endswith(rule_based_list[1]):
        return word[:word.rindex(rule_based_list[1])] + 'ी'
    
    # rule 3
    elif word.endswith(rule_based_list[2]):
        return word[:word.rindex(rule_based_list[2])] + 'ा'
      
    # rule 4
    elif word.endswith(rule_based_list[5]):
        return word[:word.rindex(rule_based_list[5])] + suffix_list[3][2]
    
    # rule 5
    elif word.endswith(rule_based_list[6]):
        return word[:word.rindex(rule_based_list[6])] + suffix_list[3][2]
    
    # rule 6
    elif word.endswith(rule_based_list[7]):
        return word[:word.rindex(rule_based_list[7])] + suffix_list[3][2]
    
    else:
        # rule 7
        if word.endswith('ा'):
            if len(syllables(word)) <= 2:
                return word
            else:
                for key in [6,5,4,3,2,1]:
                    for value in suffix_list[key]:
                        if word.endswith(value):
                            last = word.rindex(value)
                            return word[:last]
                        
        # rule 8
        elif word.endswith('ी'):
            if len(syllables(word)) <= 3:
                return word
            else:
                for key in [6,5,4,3,2,1]:
                    for value in suffix_list[key]:
                        if word.endswith(value):
                            last = word.rindex(value)
                            return word[:last]
        
        # rule 9
        else:
            for key in [6,5,4,3,2,1]:
                    for value in suffix_list[key]:
                        if word.endswith(value):
                            last = word.rindex(value)
                            return word[:last]

    return word


answer = root_word(enter_word)
print(f'मूल शब्द : {answer}')