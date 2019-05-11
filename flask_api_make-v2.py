from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import re
from textblob import Word
from gensim.models import KeyedVectors
from scipy import spatial
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import pickle
from textblob import TextBlob
from nltk import load
import pandas as pd
from nltk.corpus import stopwords
import string
import gensim
import pyemd
from pyemd import emd
import time
filename = 'glove.6B.50d.txt.word2vec'
model_gv = KeyedVectors.load_word2vec_format(filename, binary=False)
def get_vector(rw):
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model_gv.vocab.keys()):
                vec_of_vec.append(list(model_gv[wd]))
            else:
                pass
    else:
        pass
    return np.nanmean(vec_of_vec, axis = 0)#vec_of_vec


def get_similarity(rw1,rw2):
    try:
        result = 1 - spatial.distance.cosine(rw1, rw2)
    except:
        result = 0
    return result


def jaccard(list_a_str, list_b_str):
    list_a = eval(str(list_a_str))
    list_b = eval(str(list_b_str))
    intersect = set(list_a).intersection(set(list_b))
    uni = set(list_a).union(set(list_b))
    try:
        jacc = len(intersect)/len(uni)
    except:
        jacc = 0
    return jacc

tagdict = load('help/tagsets/upenn_tagset.pickle')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
lr_clf_loaded = pickle.load(open("lr_clf_wmd_new_features_v2.pickle.dat", "rb"))
app = Flask(__name__)
CORS(app, support_credentials=True)

@cross_origin(supports_credentials=True)
@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    number_of_targetCaptions = int(request.form['number_of_targetCaptions'])
    number_of_targetParagraphs = int(request.form['number_of_targetParagraphs'])
    postText = request.form['postText'].strip().replace('"',' ').replace("'",' ').replace('[',' ').replace(']',' ')
    targetTitle = request.form['targetTitle'].strip().replace('"',' ').replace("'",' ').replace('[',' ').replace(']',' ')
    targetDescription = request.form['targetDescription'].strip().replace('"',' ').replace("'",' ').replace('[',' ').replace(']',' ')
    targetParagraphs = request.form['targetParagraphs'].strip().replace('"',' ').replace("'",' ').replace('[',' ').replace(']',' ')
    targetKeywords = request.form['targetKeywords'].strip().replace('"',' ').replace("'",' ').replace('[',' ').replace(']',' ')
    targetCaptions = request.form['targetCaptions'].strip().replace('"',' ').replace("'",' ').replace('[',' ').replace(']',' ')
    di = {'number_of_targetCaptions':number_of_targetCaptions, 'number_of_targetParagraphs':number_of_targetParagraphs, 'postText':postText, 'targetTitle':targetTitle, 'targetDescription':targetDescription, 'targetParagraphs':targetParagraphs, 'targetKeywords':targetKeywords, 'targetCaptions':targetCaptions}
    #open("/data/kaggle_data_ga_crp/input_clickbait.txt", 'w').write(str(di))
    
    #Vector dimension extraction
    postText_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(postText)))]
    targetTitle_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetTitle)))]
    targetDescription_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetDescription)))]

#targetParagraphs_cleaned_splitted = [Word(j).lemmatize().strip().lower() for i in eval(str(targetParagraphs)) for j in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetParagraphs)))]
    targetParagraphs_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetParagraphs)))]

    targetKeywords_cleaned_splitted = str(targetKeywords).lower().split(',')
    targetCaptions_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetCaptions)))]
    
    #Takes maximum time
    postText_vec = get_vector(postText_cleaned_splitted)
    targetTitle_vec = get_vector(targetTitle_cleaned_splitted)
    targetDescription_vec = get_vector(targetDescription_cleaned_splitted)
    targetParagraphs_vec = get_vector(targetParagraphs_cleaned_splitted)
    targetKeywords_vec = get_vector(targetKeywords_cleaned_splitted)
    targetCaptions_vec = get_vector(targetCaptions_cleaned_splitted)
    
    postText_polarity = TextBlob(str(postText)).sentiment.polarity
    postText_subjectivity = TextBlob(str(postText)).sentiment.subjectivity
    targetCaptions_polarity = TextBlob(str(targetCaptions)).sentiment.polarity
    targetCaptions_subjectivity = TextBlob(str(targetCaptions)).sentiment.subjectivity
    targetDescription_polarity = TextBlob(str(targetDescription)).sentiment.polarity
    targetDescription_subjectivity = TextBlob(str(targetDescription)).sentiment.subjectivity
    targetParagraphs_polarity = TextBlob(str(targetParagraphs)).sentiment.polarity
    targetParagraphs_subjectivity = TextBlob(str(targetParagraphs)).sentiment.subjectivity
    targetTitle_polarity = TextBlob(str(targetTitle)).sentiment.polarity
    targetTitle_subjectivity = TextBlob(str(targetTitle)).sentiment.subjectivity

    pos_di = {}
    for pos in list(tagdict.keys()):
        pos_di[pos] = []

    di = Counter([j for i,j in pos_tag(word_tokenize(postText))])
    for pos in list(tagdict.keys()):
        pos_di[pos].append(di[pos])

    postTitle_pos_df = pd.DataFrame(pos_di)

    #number of stop words
    stp_wds = set(stopwords.words('english'))
    postText_number_of_stop_words = len(stp_wds.intersection(word_tokenize(str(postText))))

    #number of punctations
    postText_num_of_unique_punctuations = len(set(postText).intersection(set(string.punctuation)))

    #similarity : SEE ABOVE
    sim_posttext_targetTitle = get_similarity(postText_vec, targetTitle_vec)
    sim_posttext_targetDescription = get_similarity(postText_vec,targetDescription_vec)
    sim_posttext_targetParagraphs = get_similarity(postText_vec,targetParagraphs_vec)
    sim_posttext_targetKeywords = get_similarity(postText_vec,targetKeywords_vec)
    sim_posttext_targetCaptions = get_similarity(postText_vec,targetCaptions_vec)
    sim_targetTitle_targetDescription = get_similarity(targetTitle_vec, targetDescription_vec)
    sim_targetTitle_targetParagraphs = get_similarity(targetTitle_vec, targetParagraphs_vec)
    sim_targetTitle_targetKeywords = get_similarity(targetTitle_vec, targetKeywords_vec)
    sim_targetTitle_targetCaptions = get_similarity(targetTitle_vec, targetCaptions_vec)

    test_df = pd.DataFrame({'number_of_targetCaptions' : [number_of_targetCaptions], 'number_of_targetParagraphs' : [number_of_targetParagraphs], 'sim_posttext_targetDescription' : [sim_posttext_targetDescription], 'sim_posttext_targetParagraphs' : [sim_posttext_targetParagraphs], 'sim_posttext_targetKeywords' : [sim_posttext_targetKeywords], 'sim_posttext_targetCaptions' : [sim_posttext_targetCaptions], 'sim_targetTitle_targetDescription' : [sim_targetTitle_targetDescription], 'sim_targetTitle_targetParagraphs' : [sim_targetTitle_targetParagraphs], 'sim_targetTitle_targetKeywords' : [sim_targetTitle_targetKeywords], 'sim_targetTitle_targetCaptions' : [sim_targetTitle_targetCaptions], 'postText_polarity' : [postText_polarity], 'postText_subjectivity' : [postText_subjectivity], 'targetCaptions_polarity' : [targetCaptions_polarity], 'targetCaptions_subjectivity' : [targetCaptions_subjectivity], 'targetDescription_polarity' : [targetDescription_polarity], 'targetDescription_subjectivity' : [targetDescription_subjectivity], 'targetParagraphs_polarity' : [targetParagraphs_polarity], 'targetParagraphs_subjectivity' : [targetParagraphs_subjectivity], 'targetTitle_polarity' : [targetTitle_polarity], 'targetTitle_subjectivity' : [targetTitle_subjectivity], 'postText_num_of_unique_punctuations' : [postText_num_of_unique_punctuations], 'postText_number_of_stop_words' : [postText_number_of_stop_words], 'sim_posttext_targetTitle' : [sim_posttext_targetTitle]})
    test_df = pd.concat([test_df,postTitle_pos_df], axis = 1)
    
    for vec in ['postText_vec', 'targetTitle_vec', 'targetDescription_vec', 'targetParagraphs_vec', 'targetKeywords_vec', 'targetCaptions_vec']:
        try:
            for itm,dim in zip(eval(vec),range(50)):
                test_df[vec[:-4]+'_golve_dim'+str(dim)] = [itm]
        except:
            for dim in range(50):
                test_df[vec[:-4]+'_golve_dim'+str(dim)] = 0
    for cm1,cm2 in [('postText_cleaned_splitted','targetTitle_cleaned_splitted'), ('postText_cleaned_splitted', 'targetDescription_cleaned_splitted'), ('postText_cleaned_splitted', 'targetParagraphs_cleaned_splitted'), ('postText_cleaned_splitted', 'targetKeywords_cleaned_splitted'), ('postText_cleaned_splitted', 'targetCaptions_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetDescription_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetParagraphs_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetCaptions_cleaned_splitted')]:
        sns1 = ' '.join([i for i in eval(cm1)])
        sns2 = ' '.join([i for i in eval(cm2)])
        dist = model.wmdistance(sns1, sns2)
        new_df = pd.DataFrame({'wmd_'+cm1+"_"+cm2 : [dist]})
        test_df = pd.concat([test_df,new_df], axis =1)

    wh_words = set(['which','what','whose','who','whom','where','whither','whence','when','how','why','whether','whatsoever'])
    postText_has_wh_words = len(set(postText_cleaned_splitted).intersection(wh_words))

    postText_has_digits = any([i.isdigit() for i in postText])
    postText_targetTitle_jaccard = jaccard(postText_cleaned_splitted, targetTitle_cleaned_splitted)
    postText_targetDescription_jaccard = jaccard(postText_cleaned_splitted, targetDescription_cleaned_splitted)
        
    x = postText
    presence_of_click_baity_phrases = 1 if "click here" in str(x) or "exclusive" in str(x) or "won’t believe" in str(x) or "happens next" in str(x) or "you know" in str(x) or "don’t want" in str(x) else 0
        
    new_features_df = pd.DataFrame({'postText_has_wh_words':[postText_has_wh_words], 'postText_has_digits':[postText_has_digits], 'postText_targetTitle_jaccard':[postText_targetTitle_jaccard], 'postText_targetDescription_jaccard':[postText_targetDescription_jaccard], 'presence_of_click_baity_phrases':[presence_of_click_baity_phrases]})
    test_df = pd.concat([test_df, new_features_df], axis = 1)
    test_df = test_df.drop(['$',')', 'EX', 'FW', 'JJR', 'JJS', 'NNPS', 'RBR', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB', 'sim_posttext_targetKeywords', 'sim_targetTitle_targetCaptions', 'sim_targetTitle_targetKeywords', 'targetKeywords_golve_dim16', 'targetKeywords_golve_dim26', 'targetKeywords_golve_dim32', 'targetKeywords_golve_dim33', 'targetKeywords_golve_dim35','wmd_postText_cleaned_splitted_targetKeywords_cleaned_splitted', 'wmd_targetTitle_cleaned_splitted_targetKeywords_cleaned_splitted'], axis = 1)
    pred = lr_clf_loaded.predict(test_df)[0]
    is_cb = 'clickbait' if pred == 1 else 'not-clickbait'
    return is_cb
    
if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0",debug=True)    