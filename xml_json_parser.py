import pandas as pd
import xml.dom.minidom as md
from nltk.tokenize import MWETokenizer
from nltk import sent_tokenize, word_tokenize

import cv2
import numpy as np
from PIL import Image
import time
import json
import re
import wordninja
from itertools import groupby
from functools import reduce
import pandas as pd
from pathlib import Path

import spacy 
from spacy import displacy
from money_parser import price_str
import datefinder
import os
import sys
from dateutil.parser import parse

new_text_pr=[]
import string
from fuzzywuzzy import fuzz
import en_core_web_lg

start_time=time.time()
try:
    nlp = en_core_web_lg.load()
except:
    nlp = spacy.load('en_core_web_lg')
    

def omnixml_reader(path_to_xml):
    doc = md.parse(path_to_xml)
    elems = doc.getElementsByTagName('wd')
    data_xml=[]
    for elem in elems:
        x1 = int(elem.getAttribute("l"))
        y1 = int(elem.getAttribute("t"))
        x2 = int(elem.getAttribute("r"))
        y2 = int(elem.getAttribute("b"))
        if elem.firstChild.hasChildNodes():
            text=elem.firstChild.firstChild.data
        else:
            text=elem.firstChild.data
        data_xml.append([x1,y1,x2-x1,y2-y1,text])
    data=pd.DataFrame(data_xml,columns=['left','top','width','height','text'])
    return data

def image_preprocessing_lines(im,data):
    
    min_horizantal_threshol=1000
    min_vertical_pixels_threshold=5
    # print('np.array(im):',np.array(im).shape)
    img=np.array(im).astype('uint8')
    if len(img.shape)==3:
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img=img
    img=cv2.GaussianBlur(img,(5,5),0)
    img=cv2.GaussianBlur(img,(5,5),0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = 255-img

    kernel=np.ones((5,9))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel=np.ones([1,int(np.max(data['width'].values))])
    img=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel,iterations=2)

    kernel=np.ones([int(np.max(data['height'].values)),1])
    img=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel,iterations=2)
    
    
    img_row_sum = np.sum(img,axis=1).tolist()
    img_row_sum=np.array(img_row_sum)
    img_row_sum[img_row_sum<min_horizantal_threshol]=0
    img_row_sum[img_row_sum>0]=1

    img_row_v=[list(g) for k,g in groupby(img_row_sum)]
    change_done=False
    change_G=0
    for g in range(len(img_row_v)):
        if len(img_row_v[g])<min_vertical_pixels_threshold:
            if change_done is False:
                img_row_v[g]=list(np.ones(len(img_row_v[g]),dtype=int)-np.array(img_row_v[g]))

                change_done=True
                change_G=g
        if change_G+1==g:
            change_done=False

    img_row_v=reduce(lambda x,y: x+y,img_row_v)
    xdiff = np.diff(img_row_v) == 0
    xdiff_bol= xdiff.astype(np.int)
    ii = np.concatenate((np.array([0]),xdiff_bol,np.array([0])))

    lines=np.where(ii==0)[0]
    return lines


def image_text_ext(im,data,lines):
    width,height=im.size
    data_cor=[]
    for g in range(len(data['text'])):
        word=data['text'].iloc[g]
        xmin=int(data['left'].iloc[g])
        ymin=int(data['top'].iloc[g])
        xmax=xmin+int(data['width'].iloc[g])
        ymax=ymin+int(data['height'].iloc[g])

        y_center=ymin+int((ymax-ymin)/2)
        start_position=lines[np.where(lines<y_center)[0][-1]]
        data_cor.append([start_position,xmin,xmax,word])

    data_cor=np.array(data_cor,dtype='object')

    lines_text=np.unique(data_cor[:,0])
    Image_Text=[]
    for g in range(len(lines_text)):
        sss=data_cor[np.where(data_cor[:,0]==lines_text[g])[0]]
        data_list_line=sss[sss[:,1].argsort()]
        
        width_of_text=np.append([0],data_list_line[1:,1]-data_list_line[:-1,2])

        line_text_raw=''
        for i in range(len(data_list_line)):
#             print(data_list_line[i,3])
            text_width_numbers=int(width_of_text[i]/10)
            if i!=0:
                if text_width_numbers==0:
                    text_width_numbers=1
            line_text_raw=line_text_raw+text_width_numbers*' '+data_list_line[i,3]+' '
#             line_text_raw=line_text_raw+text_width_numbers*' '+' '.join(wordninja.split(data_list_line[i,3]))
        Image_Text.append(line_text_raw.replace('             ','     '))
    return Image_Text

def finding_patteren_regex(text_splited,test_str):
    regex_pattern='(?:'+text_splited[0]+'([^\n ]*) *'+text_splited[1]+' ' #Avert your eyes, it may take on other forms!

    for i in range(2,len(text_splited)):
        regex_pattern=regex_pattern+'*'+text_splited[i]+' '
    regex_pattern=regex_pattern[:-1]+')'
    
    matches = re.finditer(regex_pattern, test_str, re.MULTILINE)
    # print([regex_pattern])
    aaass=list(matches)#,aaass[-1].end()]
    if not aaass:
        # print(regex_pattern,text_splited,test_str)
        # print('na','aaass')
        return 'na'
    #return test_str[aaass[-1].start():]
    if len(aaass)>1:
        output_text=test_str[aaass[0].end():aaass[1].start()]
    else:
        output_text=test_str[aaass[-1].end():].strip()
    # output_text=test_str[aaass[-1].end():].strip()
    # print(output_text)
    if len(output_text)==0:
        return ''
    output_text=[k for k in (output_text).split(' '*7) if len(k)>2]
    if len(output_text)==0:
        return ''
    # print('output_text:',text_splited,output_text[0].strip())
    return output_text[0].replace(':','').replace('  ',' ').strip()



def spacy_extraction(Image_Text, category,company):


    #print("***************************")
    #print(text)
    #print("______________________________________")
    lst=[]
    paragraphs = [p for p in Image_Text]
    # print(paragraphs)
    """
    doc = nlp(text)     
    for ent in doc.ents: 
        print(ent.text, ent.start_char, ent.end_char, ent.label_) 
        if ent.label_ in category:
            #print(ent.text, ent.start_char, ent.end_char, ent.label_) 
            return ent.text
    """
    
    for s in paragraphs:
        s=" ".join(s.split())
        # print(s)
        doc = nlp(s)     
        for ent in doc.ents: 
            if ent.label_ in category:
                lst.append(ent.text)
    #print(lst)
    #print("#########################") 
    if not lst:
        return ""
    save_list_tags=[]
    for sub_list in lst:
        list_Saver1=[]
        for k in sub_list.split():
            if k.isnumeric()==False:
                list_Saver1.append(k)
        save_list_tags.append(list_Saver1)
    save_list_tags.sort(key=len,reverse=True)
    lst=[' '.join(k) for k in save_list_tags]
    for i in lst:
        i=''.join(ii for ii in i if not ii.isdigit())
        #print(fuzz.token_set_ratio(company.lower(), i.lower()))
        if fuzz.token_set_ratio(company.lower(), i.lower()) <90 and 'retail' not in i.lower() and ' pay' not in i.lower() and 'allowances' not in i.lower() and 'cpf' not in i.lower():
            if "jun" not in i.lower() or not any(char.isdigit() for char in i):
                i=i.replace("Apr '","")
                return i
    #return 'NA'    


def spacy_extraction_org(Image_Text, category):
    
    lst=[]
    lst_person=[]
    paragraphs = [p for p in Image_Text if p]
    # paragraphs=[g for k in paragraphs for g in k.split(' '*25)]
    # print("****************************************")
    # print(paragraphs)
    for s in paragraphs:
        # sub_para=sub_para.split(' '*15)
        # for s in sub_para:
        for i in ["ltd","limited","llp","org", "inc", "trading","traders"]:
            if i+' ' in s.lower(): 

                if "banking" in s.lower():
                    continue
                #text_split = re.split(r'\s{3,}', s)
                
                text_split=s.split(i,0)
                text_split = re.split(r'\s{5,}', s)
                #text_split="".join(s.split())
                # print(text_split)
                for st in text_split:
                    # print("****"+st)
                    
                    if i in st.lower() and "bank" not in st.lower():
                        if "ltd" in st.lower()[:3]:
                            continue
                        if ' '.join(st.lower().split())=='corporation ltd':
                            continue
                        if 'org'==st.lower().split()[0]:
                            continue
                        # print("**********st*********"+st,s)
                        
                        return st
        if "company" in s.lower():
            if "company" not in s.lower().split()[0]:
                return s

    for s in paragraphs:
        indx=paragraphs.index(s)
        s=" ".join(s.split())
        doc = nlp(s)     
        for ent in doc.ents: 
            # print(ent.text, ent.start_char, ent.end_char, ent.label_) 
            if ent.label_ in category:
                lst.append(ent) 
            if ent.label_ in "PERSON":
                if indx <2:
                    lst_person.append(ent)    
 
    # print(lst)
    # print(lst_person)
    if not lst:
        return 'na'
    for items in lst:
        
        for i in ["ltd","limited","llp","org", "inc", "trading","traders", "technologies","retail"]:
            if i in items.text.lower():
                if "banking" in items.text.lower():
                    continue
                return items.text
    
               # return s
    for items in lst_person:
        # print(str(items.start_char)+"####"+str(items.end_char)+"####")
        if items.start_char <10 and items.end_char <80 and 'pay' not in items.text.lower() and "jun" not in items.text.lower():
            if items.start_char > lst[0].start_char:
                continue    
            co_text= ''.join([i for i in items.text if not i.isdigit()])
            # print('co_text: ',[co_text])
            return co_text
            
    return lst[0].text
    

def lines_extraction(xml_path,threshold):
    doc = md.parse(xml_path)
    elems = doc.getElementsByTagName('ln')
    data_xml=[]
    coordinates=[]
    previous_y1=0
    for elem in elems:
        x1 = int(elem.getAttribute("l"))
        y1 = int(elem.getAttribute("t"))
        x2 = int(elem.getAttribute("r"))
        y2 = int(elem.getAttribute("b"))
        coordinates.append([x1,y1,x2,y2,y2-y1])
        previous_y1=y1

    coordinates=np.array(coordinates)
    coordinates=coordinates[coordinates[:,1].argsort()]

    previous_y1=0
    new_coordinates=[]
    for line in coordinates:
        new_line=list(line)+[line[1]-previous_y1]
        new_coordinates.append(new_line)
        previous_y1=line[1]

    new_coordinates=np.array(new_coordinates)

    diff=new_coordinates[:,5]
    diff[diff<int(threshold/1.5)+1]=0
    diff[diff>0]=1

    rows=[]
    for i in range(len(new_coordinates)):
        if new_coordinates[i,5]==0:
            rows.append([min(new_coordinates[i,1],new_coordinates[i-1,1]),max(new_coordinates[i,3],new_coordinates[i-1,3])])
        if new_coordinates[i,5]==1:
            rows.append([new_coordinates[i,1],new_coordinates[i,3]])

    rows=np.array(rows)

    return np.unique(rows[:,0])
       
def datefinding(text,sequence=0):
    regex_sequence=['(\d{1,2}[\/ ,.-](\d{1,2}|january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)[\/ ,.-]\d{2,4})','((january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)[\/ ,-]\s*\d{2,4})','((\d{1,2})*[\/ \-\.,]*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)[\/ ,.\- ]*\d{2,4})','((\d{1,2})*[\/ ,\-.]+\d{1,2}[\/ ,.\- ]+\d{2,4})\b','((\d{1,2})*[\/-]+\d{1,2}[\/-]+\d{2,4}[\/-]+\d{2})']#_,_,regex_patteren_date_word_number,regex_patteren_date_number
    matches = re.findall(regex_sequence[sequence],text.lower())
    matches=[m[0] for m in matches]
    return matches

def datefound_through_patteren(text_out_regex):
    text_out_regex = re.sub(':|Rs[\d,\. ]+', '|', text_out_regex, flags=re.IGNORECASE)
    text_out_regex=text_out_regex.replace('12019','/2019')
    datefound2=datefinding(text_out_regex,sequence=2)
    
    if len(datefound2)>1:
        return datefound2
    else:
        datefound0=datefinding(text_out_regex,sequence=0)
        if len(datefound0)>1:
            return datefound0
    
    if len(datefound2)>0:
        return datefound2
    if len(datefound0)>0:
        return datefound0
    else:
        return []


allowances_categories=['meal_allowance','drinks_allowance','npaa_annual_allowance','care_person_allowance','relocation_allowance','allowance_living','taxable_allowances','non_taxable_allowances','overseas_allowance','hardship_allowance','field_allowance','fbs_allowance','mobile_allowance','travel_allowance','total_allowance','other_allownces']
price_categories=['basic_pay','total_earning','net_pay']
price_categories=price_categories#+allowances_categories
all_allowances=['total_allowance','other_allownces','mobile_allowance','travel_allowance','meal_allowance','drinks_allowance','npaa_annual_allowance','care_person_allowance',
         'relocation_allowance', 'allowance_living','taxable_allowances', 'non_taxable_allowances','overseas_allowance',
            'hardship_allowance', 'field_allowance', 'fbs_allowance','other_allownces','other_allownces','flex']

            
def nlp_parser(Image_Text,mwe,dict_to_save_keys):
    Dict={}
    Allowances={}
    for s in Image_Text:
        tokenized_Sentecnces=[' '.join(sent_tokenize(s))]
        for sent in tokenized_Sentecnces:
            sent=sent.replace('(','').replace(')','').replace('*','').replace('..',' ')
            mwe_text_array=mwe.tokenize(word_tokenize(sent.lower()))
            # print(mwe_text_array)
            if '__' in ' '.join(mwe_text_array):
                for text_part_sub in mwe_text_array:
                    if '__' in text_part_sub:
                        for category in list(dict_to_save_keys.keys()):
                            if text_part_sub.split('__') in dict_to_save_keys[category]:
                                text_out_regex=finding_patteren_regex(text_part_sub.split('__'),sent.lower())
                                # print('text_out_regex:',text_out_regex)
                                if category in ['employ_id']:
                                    r = re.compile(r'(?<=\:)(\s*\w+)|$')
                                    text_out_regex=re.findall(r, text_out_regex)[0].strip()
                                
                                if category in price_categories:
                                    if '/' in sent:
                                        for dt_rep in [k for k in sent.split() if '/' in k]:
                                            sent=sent.replace(dt_rep,'')
                                    text_out_regex=finding_patteren_regex(text_part_sub.split('__'),sent.lower().replace('sgd','').replace('$',''))
                                    
                                    # print(sent.lower())
                                    if category not in Dict.keys():
                                        if bool(re.search(r'\d', text_out_regex)):
                                            text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                            
                                            if text_out_regex[0].replace(' ','')=='2019':
                                                text_out_regex=sent.lower().split('2019')[-1]
                                                text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                                # print('text_out_regextext_out_regex',text_out_regex)
                                            text_out_regex1=text_out_regex[0]
                                            try:
                                                text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                            except (ValueError):
                                                continue
                                            if len(str(int(float(str(text_out_regex1).replace(',','').replace(' ','')))))<3 and len(str(text_out_regex1).replace(' ','').replace('.',''))>0:
                                                outtxt1=sent.lower().split(text_part_sub.split('__')[-1])
                                                if len(outtxt1)>2:
                                                    outtxt1=outtxt1[1]
                                                else:
                                                    outtxt1=outtxt1[-1]
                                                    
                                                outtxt1=outtxt1.split(str(int(float(text_out_regex))))
                                                if len(outtxt1)>1:
                                                    outtxt1=outtxt1[1]
                                                else:
                                                    outtxt1=outtxt1[0]                                            
                                                if outtxt1[0]=='.' or outtxt1[0]=='0':
                                                    outtxt1=' '.join(outtxt1.split(outtxt1.split()[0])[1:])
                                                    
                                                if bool(re.search(r'\d', outtxt1)):
                                                    text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", outtxt1)[0]
                                                    if text_out_regex[0].replace(' ','')=='2019':
                                                        text_out_regex=sent.lower().split('2019')
                                                        if len(text_out_regex)>1:
                                                            text_out_regex=text_out_regex[1]
                                                        else:
                                                            text_out_regex=text_out_regex[-1]
                                                        text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                                    if text_out_regex[0].replace(' ','')=='19':
                                                        text_out_regex=sent.lower().split('19')
                                                        if len(text_out_regex)>1:
                                                            text_out_regex=text_out_regex[1]
                                                        else:
                                                            text_out_regex=text_out_regex[-1]
                                                        text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                                    
                                                    try:
                                                        text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                                    except (ValueError):
                                                        continue
                                        else:
                                            # print(category,'na')
                                            continue
                                    else:
                                        continue
                                        
                                
                                if category in ['person_name']:
                                    #spacy_text_out_regex=spacy_extraction(text_out_regex  , ['PERSON'])
                                    
                                    if any(x in text_out_regex.lower() for x in ['bank','uob','address']):
                                        continue
                                    if list(datefinder.find_dates(text_out_regex)):
                                        continue
                                        
                                    # if spacy_text_out_regex is None:
                                        # r = re.compile(r'(?<=\:)(\s*\w+\s\w*)|$')
                                        # text_out_regex=re.findall(r, text_out_regex)[0].strip()
                                    # else:
                                        # text_out_regex=spacy_text_out_regex
                                    text_out_regex = ''.join([i for i in text_out_regex if not i.isdigit()])
                                    text_out_regex =text_out_regex.replace('date join','').replace('net pay','').replace('mr ','').replace('miss ','').replace('ms. ','').replace('dr ','')
                                    if text_out_regex=='':
                                        continue
                                    
                                    if category in Dict.keys():
                                        if len(Dict[category].split())>len(text_out_regex.split()):
                                            text_out_regex=Dict[category]
                                if category in ['payment_date_month']:
                                                                    
                                    if text_out_regex is 'na':
                                        if bool(re.search(r'\d', sent.lower())): 
                                            text_out_regex = sent.lower()
                                            
                                    elif not bool(re.search(r'\d', text_out_regex)): 
                                        continue
                                    
                                    if text_out_regex is 'na':
                                        continue
                                    
                                    
                                    # print("text_out_regex_date1:",[sent.lower()],[text_out_regex],[text_part_sub])
                                    if ' to ' in text_out_regex:
                                        datefound=text_out_regex.split(' to ')
                                        if len(datefound)>1:
                                            if 'salary_start_date' not in Dict.keys():
                                                try: 
                                                    datefound[0]=datefound[0].split(' end ')[-1].replace('from','').replace('o1','01')
                                                    start_date=parse(datefound[0],dayfirst=True)
                                                    end_date=parse(datefound[1],dayfirst=True)
                                                    Dict.update({'salary_start_date' : start_date.strftime("%d/%m")+'/'+str(min(start_date.year,end_date.year))}) 
                                                    Dict.update({'salary_end_date' : end_date.strftime("%d/%m")+'/'+str(min(start_date.year,end_date.year))})
                                                except:
                                                    continue
                                    
                                    if 'salary_start_date' not in Dict.keys():
                                        if text_part_sub=='month__:':
                                            datefound=[]
                                            if ' to ' in text_out_regex:
                                                datefound=text_out_regex.split(' to ')
                                            if '-' in text_out_regex:
                                                datefound=text_out_regex.split('-')
                                                if len(datefound)==2:
                                                    try: 
                                                        datefound[0]=datefound[0].split(' end ')[-1].replace('from','').replace('o1','01')
                                                        start_date=parse(datefound[0],dayfirst=True)
                                                        end_date=parse(datefound[1],dayfirst=True)
                                                        Dict.update({'salary_start_date' : start_date.strftime("%d/%m")+'/'+str(min(start_date.year,end_date.year))}) 
                                                        Dict.update({'salary_end_date' : end_date.strftime("%d/%m")+'/'+str(min(start_date.year,end_date.year))})
                                                    except:
                                                        continue
                                                    
                                    
                                    datefound=datefound_through_patteren(text_out_regex)
                                    if len(datefound)==0:
                                        continue
                                    try:
                                        if 'salary_start_date' not in Dict.keys():
                                            if len(datefound)>1:
                                                Dict.update({'salary_start_date' : parse(datefound[0].replace(',','.'),dayfirst=True).strftime("%d/%m/%Y")}) 
                                                Dict.update({'salary_end_date' : parse(datefound[1].replace(',','.').replace('  ',''),dayfirst=True).strftime("%d/%m/%Y")})
                                                
                                        if 'payment_date_month' not in Dict.keys():
                                            if len(datefound)>0:
                                                text_out_regex = parse(datefound[0].replace(',','.'),dayfirst=True).strftime("%m/%Y")
                                        else:
                                            continue
                                    except:
                                        continue
                                    
                                # print(category)
                                if category in ['company_name']:    
                                    # print(text_out_regex)
                                    if "company" in text_out_regex.lower():
                                        text_split = re.split(r'\s{3,}', text_out_regex)
                                        
                                        for st in text_split:
                                            if "company" not in st.lower():
                                                text_out_regex= st
                                    # print(text_out_regex)
                                if category in all_allowances:
                                    # print('first',category,[text_out_regex],'len:',len(text_out_regex.split('.')))
                                    if text_out_regex[0:2]=='. ':
                                        text_out_regex=text_out_regex[2:]
                                    text_out_regex=text_out_regex.split('  ')[0]
                                    if len(text_out_regex.split('.'))>2:
                                        sent=sent.replace(text_out_regex,'')
                                        text_out_regex=finding_patteren_regex(text_part_sub.split('__'),sent.lower().replace('sgd','').replace('$',''))
                                    
                                    
                                    
                                    if '/' in text_out_regex:
                                        #print('-------------------------------')
                                        #print([text_out_regex])
                                        # print('##########################',[k for k in text_out_regex.split() if '/' in k])
                                        for dt_rep in [k for k in text_out_regex.split() if '/' in k]:
                                            sent=sent.replace(dt_rep,'')
                                        text_out_regex=finding_patteren_regex(text_part_sub.split('__'),sent.lower().replace('sgd','').replace('$',''))
                                        #print([text_out_regex])
                                    
                                    
                                    # text_out_regex_ref=re.findall("([0-9]+[,.]+[0-9]+)|$", text_out_regex)[0]
                                    text_out_regex_ref=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0][0]
                                    # print('second1:',category,[text_out_regex],[text_out_regex_ref])
                                    
                                    if text_out_regex_ref=='2019':
                                        for sub_text_out_regex in text_out_regex.split():
                                            # print(sub_text_out_regex)
                                            sent=sent.lower().replace(sub_text_out_regex.lower(),'')
                                        # print('sent:',[sent])
                                        text_out_regex=finding_patteren_regex(text_part_sub.split('__'),sent.lower().replace('sgd','').replace('$',''))
                                        text_out_regex_ref=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0][0]
                                        # print('second2:',category,[text_out_regex],[text_out_regex_ref])
                                    
                                    # print('second:',category,text_out_regex_ref)
                                    if text_out_regex_ref is "" or not text_out_regex_ref:
                                        text_out_regex=re.findall("([\$]*[\s]*[0-9]+[,.]*[0-9]*)|$", text_out_regex)[0]
                                    else:
                                        text_out_regex=text_out_regex_ref
                                    # print('third:',category,text_out_regex)
                                    for ch in [",","-","$"," "]:
                                        text_out_regex=text_out_regex.replace(ch,"")
                                    text_out_regex=text_out_regex.strip()
                                    
                                    if not bool(re.search(r'\d', text_out_regex)):
                                        continue
                                    # print('all_allowances_start:',[text_out_regex])
                                    if float(text_out_regex)==0:
                                        continue
                                    duplicate=False
                                    for i in Allowances.keys():                         
                                        if text_part_sub.replace("__","_") in i and float(text_out_regex)==float(Allowances[i]) :# in ["0.00"]:
                                            #print("duplicate cuaght",value.replace("__",""),text_out_regex)
                                            duplicate=True
                                    if float(float(text_out_regex)-int(float(text_out_regex)))==0.0: 
                                            
                                            text_out_regex=str(float(text_out_regex))
                                            if text_out_regex.split('.')[-1]=='0':
                                                text_out_regex=str(int(float(text_out_regex)))
                                    if not duplicate:
                                        # print("text_part_sub:",text_part_sub.replace("__","_") ,text_out_regex)
                                        Allowances.update({text_part_sub.replace("__","_") : text_out_regex})
                                    continue
                                
                                if category in ['salary_start_date']:
                                    text_out_regex=text_out_regex.split('to')[0]
                                    try:
                                        text_out_regex=parse(text_out_regex.replace('  ',' '),dayfirst=True).strftime("%d/%m/%Y")
                                    except:
                                        continue
                                if category in ['salary_end_date']:
                                    text_out_regex=text_out_regex.split('to')[-1]
                                    try:
                                        text_out_regex=parse(text_out_regex,dayfirst=True).strftime("%d/%m/%Y")
                                    except:
                                        continue
                                Dict.update({category : text_out_regex.title().strip()})

            if 'salary_start_date' not in Dict.keys():
                if 'salary_end_date' not in Dict.keys():
                    datefound=datefinding(sent.replace('12019','/2019'),sequence=0)
                    if len(datefound)==2:
                        # print(' '.join(datefound))
                        
                        if not any(x in ' '.join(datefound) for x in ['.00','-00',' 00','/00','..']):                
                            try:                                        
                                Dict.update({'salary_start_date' : parse(datefound[0],dayfirst=True).strftime("%d/%m/%Y")}) 
                                Dict.update({'salary_end_date' : parse(datefound[1],dayfirst=True).strftime("%d/%m/%Y")})
                            except (ValueError):
                                pass
                    else:
                        datefound=datefinding(sent,sequence=2)
                        if 'salary_start_date' not in Dict.keys():
                            if len(datefound)>1:
                                Dict.update({'salary_start_date' : parse(datefound[0],dayfirst=True).strftime("%d/%m/%Y")}) 
                                Dict.update({'salary_end_date' : parse(datefound[1].replace('  ',''),dayfirst=True).strftime("%d/%m/%Y")})


     #   df[image_name].append(Dict)
      ##  df([Dict])
     #   doc = nlp(s)
     #   print('Spacy NER:')                        
     #   for ent in doc.ents:
     #       print(ent.text, ent.start_char, ent.end_char, ent.label_) 
    if 'salary_start_date'  not in Dict.keys():
        for s in Image_Text:
            tokenized_Sentecnces=[' '.join(sent_tokenize(s))]
            for sent in tokenized_Sentecnces:
                datefound=datefinding(sent,sequence=1)
                if len(datefound)==2:
                    Dict.update({'salary_start_date' : '01/'+parse(datefound[0],dayfirst=True).strftime("%m/%Y")}) 
                    Dict.update({'salary_end_date' : '01/'+parse(datefound[1],dayfirst=True).strftime("%m/%Y")})
                if len(datefound)==1:
                    if 'payment_date_month'  not in Dict.keys():
                        Dict.update({'payment_date_month' : parse(datefound[0],dayfirst=True).strftime("%m/%Y")})
                
    
    for s in Image_Text:
        if 'salary_start_date'  not in Dict.keys():
            if 'salary_end_date'  not in Dict.keys():
                
                tokenized_Sentecnces=[' '.join(sent_tokenize(s))]
                for sent in tokenized_Sentecnces:
                    datefound=datefinding(sent,sequence=4)
                    if len(datefound)==1:
                        datefound=datefound[0].split('-')
                        Dict.update({'salary_start_date' : parse(datefound[0],dayfirst=True).strftime("%d/%m/%Y")}) 
                        Dict.update({'salary_end_date' : parse(datefound[1],dayfirst=True).strftime("%d/%m/%Y")})
    
    
    
    if 'salary_start_date'  not in Dict.keys():
        Dict.update({'salary_start_date' : ""})      
    
        
    if 'salary_end_date'  not in Dict.keys():
        Dict.update({'salary_end_date' : ""})     
    if Dict['salary_start_date']!='':
        if Dict['salary_end_date']!='':
            Dict.update({'payment_date_month' : parse(Dict['salary_start_date'],dayfirst=True).strftime("%m/%Y")})
    
    if 'payment_date_month'  not in Dict.keys():
        Dict.update({'payment_date_month' : ""})
 
    if Dict["salary_start_date"] is not "" and  Dict["salary_end_date"] is not "":
        if Dict["salary_start_date"][3:5] is not Dict["salary_end_date"][3:5]:
            if Dict["salary_start_date"][:2] in Dict["salary_end_date"][3:5]:
                
                month =Dict["salary_start_date"][:2]
                day= Dict["salary_start_date"][3:5]
                new_date=Dict["salary_start_date"]
                new_date = day +'/'+month+ '/'+new_date[5 + 1:]
                Dict["salary_start_date"]=new_date
                
            elif Dict["salary_end_date"][:2] in  Dict["salary_start_date"][3:5]:
                month =Dict["salary_end_date"][:2]
                day= Dict["salary_end_date"][3:5]
                new_date=Dict["salary_end_date"]
                new_date = day +'/'+month+ '/'+new_date[5 + 1:]
                Dict["salary_end_date"]=new_date
    if Dict["salary_start_date"] is not "":
        if int(Dict["salary_start_date"][6:]) < 1990:
            Dict["salary_start_date"]=""
    if Dict["salary_end_date"] is not "":
        if int(Dict["salary_end_date"][6:]) < 1990:
            Dict["salary_end_date"]=""
    
    if Dict["payment_date_month"] is not "":
        if int(Dict["payment_date_month"][3:]) < 1990:
            if Dict['salary_end_date'] not in "":
                Dict["payment_date_month"]=Dict["salary_end_date"][3:]
    if Dict["salary_start_date"] is not "" and Dict["salary_end_date"] is not "":
        if Dict["salary_start_date"][6:] is not Dict["salary_end_date"][6:]:
            if Dict["payment_date_month"] is not "":
                if int(Dict["payment_date_month"][3:]) == int(Dict["salary_start_date"][6:]):
                     #change end date year
                    year =Dict["payment_date_month"][3:]
                
                    new_date=Dict["salary_end_date"]
                    new_date = new_date[:5] +'/'+ year
                    Dict["salary_end_date"]=new_date
                elif int(Dict["payment_date_month"][3:]) == int(Dict["salary_end_date"][6:]):
                    #change start date year
                    year =Dict["payment_date_month"][3:]
                
                    new_date=Dict["salary_start_date"]
                    new_date = new_date[:5] +'/'+ year
                    Dict["salary_start_date"]=new_date
    for sent in Image_Text:
        sent=sent.lower()
        for category in list(dict_to_save_keys.keys()):
            if category in ['basic_pay']:
                if category not in Dict.keys():
                    total_values=[]
                    for value in dict_to_save_keys[category]:
                        if len(value)<2:
                            total_values.append(value[0])
                    for value in total_values:
                        sent=sent.replace(value,value+'__')
                        text_out_regex=finding_patteren_regex([value,' '],sent.lower())
                        # print('text_out_regex::::',text_out_regex)
                        if text_out_regex!='na':
                            if text_out_regex!='':
                                if bool(re.search(r'\d', text_out_regex)):
                                    text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                
                                    if text_out_regex[0]=='2019':
                                        text_out_regex=sent.lower().split('2019')[-1]
                                        text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                        # print('text_out_regextext_out_regex',text_out_regex)
                                    try:
                                        text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                    except (ValueError):
                                        continue
                                    
                                    Dict.update({category : text_out_regex.replace(',','')})
            if category not in ['payment_date_month', 'basic_pay', 'total_earning', 'net_pay', 'employ_id']:
                if category not in Dict.keys():
                    total_values=[]
                    for value in dict_to_save_keys[category]:
                        if len(value)<2:
                            total_values.append(value[0])
                    for value in total_values:
                        if value=='company':
                            if sent.split()[0]!=value:
                                continue
                            if sent.split()[1] in ['stomp','stamp','message','reg','nix!']:
                                continue
                            
                        sent=sent.replace(value,value+'__')
                        # print(sent)
                        text_out_regex=finding_patteren_regex([value,' '],sent.lower())
                        if text_out_regex!='na':
                            if text_out_regex!='':
                                if category=='person_name':
                                    #avoid_list=["bank","uob","address","emp no"]
                                    if text_out_regex.isnumeric():
                                        text_out_regex=finding_patteren_regex([value,' '],sent.lower().replace(text_out_regex,''))
                                        if text_out_regex.isnumeric():
                                            text_out_regex=finding_patteren_regex([value,' '],sent.lower().replace(text_out_regex,''))
                                    
                                    if "bank" not in text_out_regex.lower() and "uob" not in text_out_regex.lower() and "address" not in text_out_regex.lower() and "emp no" not in text_out_regex.lower() :  
                                        # print(text_out_regex.title())
                                        text_out_regex = ''.join([i for i in text_out_regex if not i.isdigit()])
                                        text_out_regex=text_out_regex.lower()
                                        text_out_regex =text_out_regex.replace('date join','').replace('net pay','').replace('mr ','').replace('miss ','').replace('dr ','').replace('/','').replace('ms. ','')
                                        if text_out_regex.lower()=='of employer' or text_out_regex.lower()=='of employee' or text_out_regex=='':
                                            continue
                                        
                                        Dict.update({category : text_out_regex.title().strip()})
                                        continue
                                d=0
                                for c in text_out_regex:
                                    if c.isdigit():
                                        d=d+1  
                                
                                if category=='company_name' and d>7:
                                    continue
                                
                                else :
                                    if category in all_allowances:
                                        if '/' in text_out_regex:
                                            for dt_rep in [k for k in text_out_regex.split() if '/' in k]:
                                                text_out_regex = text_out_regex.replace(dt_rep, '')
                                                sent=sent.replace(dt_rep,'')
                                                sent=sent.replace(value,value+'__')
                                                text_out_regex=finding_patteren_regex([value,' '],sent.lower())
                                                
                                        if not bool(re.search(r'\d', text_out_regex)):
                                            continue
                                        
                                        text_out_regex_ref=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0][0]
                                        
                                        
                                        
                                        if text_out_regex_ref=='2019':
                                            for sub_text_out_regex in text_out_regex.split():
                                                sent=sent.lower().replace(sub_text_out_regex.lower(),'')
                                            sent=sent.replace(value,value+'__')
                                            text_out_regex=finding_patteren_regex([value,' '],sent.lower())
                                            
                                            text_out_regex_ref=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0][0]
                                            # print('second2(820):',category,[text_out_regex],[text_out_regex_ref])
                                            
                                        # print('text_out_regex_ref(808):',text_out_regex_ref)
                                        if text_out_regex_ref is "" or not text_out_regex_ref:
                                            text_out_regex_ref=re.findall("([\$]*[\s]*[0-9]+[,.]+[0-9]*)|$", text_out_regex)[0]
                                
                                            if text_out_regex_ref is "" or not text_out_regex_ref:
                                                text_out_regex_ref=text_out_regex_ref=re.findall("([\$]*[\s]*[0-9]+[,.]*[0-9]*)|$", text_out_regex)[0]
                                        
                                        text_out_regex_ref=text_out_regex_ref.split()
                                        if len(text_out_regex_ref)>1:
                                            text_out_regex_ref=''.join(text_out_regex_ref[0:2])
                                        else:
                                            text_out_regex_ref=text_out_regex_ref[0]
                                        for ch in [",","-","$"]:
                                            text_out_regex_ref=text_out_regex_ref.replace(ch,"")
                                        text_out_regex=text_out_regex_ref.strip()  
                                        duplicate=False
                                        zero_value=False
                                        
                                        if len(text_out_regex.split('.'))>2:
                                            continue
                                        
                                        for i in Allowances.keys():
                                            if value.replace("__","").lower() in i.lower() and float(text_out_regex)==float(Allowances[i]):
                                               #print("duplicate cuaght",value.replace("__",""),text_out_regex)
                                                duplicate=True
                                            
                                        if float(text_out_regex)==0:
                                            zero_value=True
                                            continue
                                        if float(float(text_out_regex)-int(float(text_out_regex))) ==0.0: 
                                            text_out_regex=str(int(float(text_out_regex))) 
                                        if not duplicate and not zero_value:
                                            Allowances.update({value.replace("__",""):text_out_regex})
                                    else:
                                        if category=='person_name':
                                            #avoid_list=["bank","uob","address","emp no"]
                                            if "bank" not in text_out_regex.lower() and "uob" not in text_out_regex.lower() and "address" not in text_out_regex.lower() and "emp no" not in text_out_regex.lower() :
                                                Dict.update({category : text_out_regex.title()})
                                        else:
                                            Dict.update({category : text_out_regex.title()})
    # if 'company_name' not in Dict.keys(): 
        # text=spacy_extraction_org(Image_Text , ['ORG']) 
        # Dict.update({'company_name' : text.title()}) 
    if 'company_name' not in Dict.keys() or not any(c.isalpha() for c in Dict["company_name"]) or Dict["company_name"].lower() is "ltd": 
       # listToStr = ' '.join(map(str, Image_Text)) 
        text=spacy_extraction_org(Image_Text , ['ORG']) 
        text=text.split(' PERIOD ')[0]
        # text=text.split(' '*15)[0]
        for rpl_word in ['Payslip   for','Payslip for',' PAYSLIP','PAYSLIP',' Payslip','Salary  Advice:','Salary  Advice','PAY  ADVICE  FOR','Pay Date']:
            text=text.replace(rpl_word,'')    
        # text=text.replace('Payslip   for','').replace('Payslip for','').replace(' PAYSLIP','').replace('PAYSLIP','').replace(' Payslip','').replace('Salary  Advice:','').replace('Salary  Advice','')
        
        # print('---------------------------')
        # print([text])
        Dict.update({'company_name' : text.strip().title()}) 
    
    '''
    company_extra_words=['co name','co  name','co   name','company name','company  name','company   name', 'c name','c   name', 'pay entity','pay  entity','pay   entity',"approved by","approved  by","approved   by","business  title"]
    

    big_regex = re.compile('|'.join(map(re.escape, company_extra_words)))
    Dict["company_name"] = big_regex.sub("", str(Dict["company_name"]).lower())
    '''
    
    company_savor0=[]
    for i in ["ltd","limited","llp","org", "inc", "trading","(singapore)","singapore"]:
        if i in str(Dict["company_name"]).lower() :
            if Dict["company_name"].lower().split()[0]!=i:
                comp_text=" ".join(str(Dict["company_name"]).lower().split())
                company_savor0.append((comp_text.split(i)[0]+i).split())
    if len(company_savor0)>0:
        company_savor0.sort(key=len,reverse=True)
        out_text=' '.join(company_savor0[0])
        if out_text[0]=='.' or out_text[0]==':':
            out_text=out_text[1:]
        Dict["company_name"]=out_text.strip()
    # Dict["company_name"]=Dict["company_name"].translate(str.maketrans('', '', punc)).title()
    
    '''
    if 'person_name' not in Dict.keys(): 
        text=spacy_extraction(" ".join(Image_Text)  , ['PERSON']) 
        if text is None:
            text=''
        Dict.update({'person_name' : text.title()})    
    '''
    
    # print(Dict,'+++++++++++++++++++++')
    if 'person_name' not in Dict.keys() or Dict["person_name"] is None: 
        
        text=spacy_extraction(Image_Text  , ['PERSON'],Dict['company_name']) 
        # print(text)
        if (text is None) or ('income' in text.lower()):
            text=""
        Dict.update({'person_name' : text.title()})
    '''
    if 'person_name_handling' not in Dict.keys(): 
        text=spacy_extraction(' '.join(map(str, Image_Text))  , ['PERSON']) 
        Dict.update({'person_name_handling1' : text})
    if 'company_name_handling' not in Dict.keys(): 
        text=spacy_extraction(' '.join(map(str, Image_Text))  , ['ORG']) 
        Dict.update({'company_name_handling1' : text})
    ''' 
    #Dict.update({'FileName' : image_name})
    # print(Allowances)
    local_company=re.sub(' +', ' ', Dict["company_name"])
    rep = {"singa pore": "Singapore","pteltd": " Pte Ltd","Rte Ltd":"Pte Ltd", "Ple Ltd": "pte Ltd", "pleltd":" pte ltd","trading":" trading", "Strictly Private Confidential" : "","Dept":"","Department":"","singapore":" Singapore "}
    for items in rep.keys():
        insensitive_hippo = re.compile(re.escape(items), re.IGNORECASE)
        local_company= insensitive_hippo.sub(rep[items], local_company)

    # rep = dict((re.escape(k), v) for k, v in rep.items()) 
    # pattern = re.compile("|".join(rep.keys()),re.IGNORECASE)
    # local_company = pattern.sub(lambda m: rep[re.escape(m.group(0))], local_company)

    local_company=re.sub(' +', ' ', local_company)
    Dict["company_name"]=local_company.title().strip()
    #Dict["company_name"]=Dict["company_name"].translate(str.maketrans('', '', punc))
    
    # print(Allowances)
    for category in list(Allowances.keys()):
        if int(float(Allowances[category]))<2:
            Allowances.pop(category)
          
    Dict.update({"allowances" : Allowances})
    
    
    if len(Dict["company_name"])<4:
        Dict["company_name"]=''
        
            # print(mwe_text_array)
    
    # print(Dict)
    for s in Image_Text:
        tokenized_Sentecnces=[' '.join(sent_tokenize(s.lower()))]
        for sent in tokenized_Sentecnces:
            sent=sent.replace('(','').replace(')','').replace('*','')
            mwe1 = MWETokenizer([["salaries","&","wages"],["monthly","salary"],["gross","salary"]], separator='__')
            mwe_text_array=mwe1.tokenize(word_tokenize(sent.lower()))
            if '__' in ' '.join(mwe_text_array):
                for text_part_sub in mwe_text_array:
                    if '__' in text_part_sub:
                        for category in list(dict_to_save_keys.keys()):
                            if category in ['basic_pay']:
                                # print('cate:',category)
                                if (category not in Dict.keys()) or (Dict[category]=='') or (len(Dict[category].replace('.',''))<3):
                                    # print('1st',text_part_sub)
                                    text_out_regex=finding_patteren_regex(text_part_sub.split('__'),sent.lower())
                                    
                                    if bool(re.search(r'\d', text_out_regex)):
                                        text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                        if text_out_regex[0].replace(' ','')=='2019':
                                            text_out_regex=sent.lower().split('2019')[-1]
                                            text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                        
                                        if text_out_regex[0].replace(' ','')=='19':
                                            text_out_regex=sent.lower().split('19')[-1]
                                            # print('after text:',text_out_regex)
                                            text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                        text_out_regex1=text_out_regex[0]
                                        
                                        try:
                                            text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                        except (ValueError):
                                            continue
                                        
                                        if len(str(int(float(str(text_out_regex1).replace(',','').replace(' ','')))))<3 and len(str(text_out_regex1).replace(' ','').replace('.',''))>0:
                                            outtxt1=sent.lower().split(text_part_sub.split('__')[-1])
                                            if len(outtxt1)>2:
                                                outtxt1=outtxt1[1]
                                            else:
                                                outtxt1=outtxt1[-1]
                                            outtxt1=outtxt1.split(str(int(float(text_out_regex))))
                                            if len(outtxt1)>1:
                                                outtxt1=outtxt1[1]
                                            else:
                                                outtxt1=outtxt1[0]                                            
                                            if bool(re.search(r'\d', outtxt1)):
                                                text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", outtxt1)[0]
                                                
                                                if text_out_regex[0].replace(' ','')=='2019':
                                                    text_out_regex=sent.lower().split('2019')
                                                    if len(text_out_regex)>1:
                                                        text_out_regex=text_out_regex[1]
                                                    else:
                                                        text_out_regex=text_out_regex[-1]
                                                    text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                                if text_out_regex[0].replace(' ','')=='19':
                                                    text_out_regex=sent.lower().split('2019')
                                                    if len(text_out_regex)>1:
                                                        text_out_regex=text_out_regex[1]
                                                    else:
                                                        text_out_regex=text_out_regex[-1]
                                                    text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                                try:
                                                    text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                                except (ValueError):
                                                    continue
                                                
                                            Dict.update({category : text_out_regex.title().strip()})
                                        Dict.update({category : text_out_regex.title().strip()})
                                        # print(Dict)
                                    else:
                                        continue
                                else:
                                    continue
    
    
    
    currency_dict = {'sgd':0,'$':0}
    
    for sent in Image_Text:
        sent1=[sent].copy()
        sent=sent.lower()
        if 'sgd' in sent:
            currency_dict['sgd']=currency_dict['sgd']+1
        if '$' in sent:
            currency_dict['$']=currency_dict['$']+1
        for category in list(dict_to_save_keys.keys()):
            if category in ['basic_pay']:
                if (category not in Dict.keys()) or (Dict[category]=='') or (len(str(int(float(Dict[category]))))<3):
                    for value in ["salary","basic","earning"]:
                        # sent=sent.replace(value,value+'__')
                        text_out_regex=finding_patteren_regex([value,' '],sent.replace(value,value+'__'))
                        # print('text_out_regex::::',text_out_regex)
                        if text_out_regex!='na':
                            if text_out_regex!='': 
                                if bool(re.search(r'\d', text_out_regex)):
                                    text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                    if text_out_regex[0]=='2019':
                                        text_out_regex=sent.lower().split('2019')[-1]
                                        text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                        # print('text_out_regextext_out_regex',text_out_regex)
                                    try:
                                        text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                        text_out_regex=text_out_regex.replace(',','')
                                        if len(text_out_regex)>9 or len(str(int(float(str(text_out_regex).replace(',','').replace(' ','')))))<3 or str(int(float(str(text_out_regex).replace(',','').replace(' ',''))))=='2018' or str(int(float(str(text_out_regex).replace(',','').replace(' ',''))))=='2019':
                                            continue
                                        Dict.update({category : text_out_regex})
                                    except (ValueError):
                                        continue

    
                    for value in ['mth salary','gross earning',"basic salary"]:
                        if value in ' '.join(sent.split()):
                            # if 'ytd '+value in ' '.join(sent.split()):
                                # continue
                            text_out_regex=Image_Text[Image_Text.index(sent1[0])+1]
                            if text_out_regex!='': 
                                if bool(re.search(r'\d', text_out_regex)):
                                    text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                
                                    if text_out_regex[0]=='2019':
                                        text_out_regex=sent.lower().split('2019')[-1]
                                        text_out_regex=re.findall("((\d+)(([., ]|\d+|)(\d+|)([., ]|\d+)(\d+|)))|$", text_out_regex)[0]
                                        # print('text_out_regextext_out_regex',text_out_regex)
                                    try:
                                        text_out_regex=str(float(text_out_regex[0].replace(',','').replace(' ','')))
                                    except (ValueError):
                                        continue
                                    text_out_regex=text_out_regex.replace(',','')
                                    if len(text_out_regex)>9:
                                        continue
                                    Dict.update({category : text_out_regex})
    
    from operator import itemgetter
    var_text, value = max(currency_dict.items(), key=itemgetter(1)) 
    Dict.update({'currency' : var_text})
    return Dict



def xml_payslip_json(uuid_name,handling_keys='handling_keys.json',extra_identity_name='_data',image_folder='preprocessed_images',xml_folder='ocr_output',jsons_folder='jsons'):
    '''
    Extraction data from ocr xml based on keypoints.
    
    Parameters:
    uuid_name (str): specific encrypted uuid_name used to save image,xml.
    
    
    
    Data Reading:
    image_path=image_folder+'/'+uuid_name+'_data.jpg'
    xml_path=xml_folder+'/'+uuid_name+'_data.xml'
    outputjson_path=jsons_folder+'/'+uuid_name+'_data.json'
    
    Returns:
    dictionary:Returning dictionary of datafound in xml based on keypoints.
    '''
    
    # print(uuid_name)
    
    # global nlp
    # start_time=time.time()
    # try:
        # nlp = en_core_web_lg.load()
    # except:
        # nlp = spacy.load('en_core_web_lg')
    
    global punc
    punc=string.punctuation

    
    if type(xml_folder)==list:
        xml_path=(xml_folder[0]).replace('0.xml','data.xml')
    else:
        xml_path=(xml_folder+'/'+uuid_name+extra_identity_name+'.xml').replace('.xml','_data.xml').replace('1.xml','1_data.xml').replace('2.xml','2_data.xml').replace('3.xml','3_data.xml')
    if  image_folder=='':
        image_path=image_folder+uuid_name+extra_identity_name+'.jpg'
    else:
        image_path=image_folder+'/'+uuid_name+extra_identity_name+'.jpg'
    outputjson_path=jsons_folder+'/'+uuid_name+extra_identity_name+'.json'
    outputjson_path=Path(__file__).parent.joinpath(outputjson_path)
    handling_keys=Path(__file__).parent.joinpath(handling_keys)
    if os.path.isfile(sys.argv[1]):
        pass
    else:
        image_path=image_path.replace('.jpg','_preprocessed.jpg')
    # print(image_path,image_folder)
    with open(handling_keys,'r') as f:
        dict_to_save_keys=json.load(f)
    tags=[]
    for cate in list(dict_to_save_keys.keys()):
        tags=tags+dict_to_save_keys[cate]
    mwe = MWETokenizer(tags, separator='__')
    im = Image.open(image_path)
    
    # im_save_path='/home/ubuntu/ocbc/storage/app/public/payslips/'+uuid_name+extra_identity_name+'.jpg'
    # im.save(im_save_path)
    
    data=omnixml_reader(xml_path)
    
    if len(data)>0:
        
        threshold=np.mean(data['height'].values)
        lines=lines_extraction(xml_path,threshold)  #image_preprocessing_lines(im,data)
        Image_Text=image_text_ext(im,data,lines)
        # print(Image_Text)
        with open ('omniocr_text/'+uuid_name+'.txt','w',encoding="utf-8") as f:
            f.write('\n'.join(Image_Text))
        
        Dict=nlp_parser(Image_Text,mwe,dict_to_save_keys)
        Dict.update({'timestamp' : round(time.time()-start_time,3)})
        for items in dict_to_save_keys.keys():
            if items not in Dict.keys() and items not in all_allowances:
                Dict.update({items : ""})
            if "allowances" not in Dict.keys():
                Dict.update({"allowances" : ""})
        
        # Allowances={}
        # Sorted_dict={}

        for key in list(Dict.keys()):
            if Dict[key]=='na':
                Dict[key]=''
            if Dict[key]=='Na':
                Dict[key]=''
            '''
            if key in all_allowances:
                Allowances[key]=Dict[key].title()
            else:
                Sorted_dict[key]=Dict[key]
                    
        Sorted_dict['allowances']=Allowances
            '''
        with open(outputjson_path,'w') as f:
            str = json.dumps(Dict, indent=4)
            f.write(str)
        return Dict



def dictionary_update(new_dictionary,handling_keys_path='handling_keys.json'):
    '''
    Update the current dictionary
    
    Input:
    new_dictionary: new dictionary consists of new keys or new variable of old keys of dictionary.
    
    handling_keys_path: (default:handling_keys.json) path of handling keys file.
    
    Save the updated dictionary as handling_keys.json
    '''
    with open(handling_keys_path,'r') as f:
        dict_to_save_keys=json.load(f)
    
    for key in list(new_dictionary.keys()):
        if key in list(dict_to_save_keys.keys()):
            dict_to_save_keys[key]=[list(new_l) for new_l in list(set([tuple(k) for k in dict_to_save_keys[key]+new_dictionary[key]]))]
        if key not in list(dict_to_save_keys.keys()):
            dict_to_save_keys[key]=new_dictionary[key]
    
    with open(handling_keys_path,'w') as f:
        json.dump(dict_to_save_keys,f)
    