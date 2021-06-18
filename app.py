import streamlit as st
import pandas as pd
import pybase64
import nltk
import sklearn
from nltk.corpus import wordnet
import joblib
tokenizer = nltk.RegexpTokenizer('\w+')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import ast

df_tags_topics = pd.read_csv('df_tags_topics.csv').copy()

lemmatizer = WordNetLemmatizer()


st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:500;
        font-size:30px;
        color: #ffffff;
        padding-top: 5px;
    }
    .logo-img {
        float:right;
        width:100px;
        height:100px
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <p class="logo-text">Openclassrooms - Projet 5<br>
        Parcours Ingénieur Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)





st.markdown("Suggestion automatique de tags pour votre post StackOverflow",
           unsafe_allow_html=True)





user_input1 = st.text_input("Copier le titre de votre post stack Overflow")
user_input2 = st.text_area("Copier le corps de votre post stack Overflow", height=200)
user_input = user_input1 + ' ' +user_input2

ButtonON = st.button("Analyser le texte")


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def sumListStr(Lwords):
    string=Lwords[0]
    for word in Lwords[1:]:
        string = string + ' ' + word
    return string

tokenizer = nltk.RegexpTokenizer('\w+')

def modification(text):
    text = text.lower()
    text_tokens = tokenizer.tokenize(text)
#     Ldelete = stopwords.words('english') + ['wants', 'want', 'wanted'] + word.isdigit()
#     text_tokens = [word for word in text_tokens if word not in Ldelete]
    
    text_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text_tokens]
    textfinal = sumListStr(text_tokens)
    return textfinal

def modificationTokenizer(text):
    text = text.lower()
    text_tokens = tokenizer.tokenize(text)
#     Ldelete = stopwords.words('english') + ['wants', 'want', 'wanted'] + word.isdigit()
#     text_tokens = [word for word in text_tokens if word not in Ldelete]
    
    text_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text_tokens]
    return text_tokens

def Affichage(Yf):
    txt = ''
    for i in Yf[0] :
        txt += i
        txt += ' '
    return txt
# if result:
#     if len(user_input2) !=0 :
#         if len(user_input1) ==0 : 
#             st.markdown("<span style='color:orange'>Conseil : Vous pouvez rajouter le titre du post pour avoir une prédiction plus fiable</span>",unsafe_allow_html=True)
#         st.markdown("Les tags suivants pourraient correspondre à votre post :")
#         st.markdown('**'+user_input+'**',unsafe_allow_html=True)
#         text_tokens = modification(user_input)
#         st.markdown(text_tokens, unsafe_allow_html=True)
        
#     else :
#         st.markdown("<span style='color:red'>Analyse impossible : Vous devez entrer au moins le corps du post pour avoir des suggestions de tags</span>",unsafe_allow_html=True)

# Real



ModelSup = joblib.load('modelSup.joblib')
SVD = joblib.load('TruncatedSVD_model.joblib')
TFIDF = joblib.load('tfidfmodel.joblib')
mlb = joblib.load("mlb.joblib")
lda_model = joblib.load("lda_model.joblib")
id2word = joblib.load("id2word.joblib")

if ButtonON:
    text = modification(user_input)
    if len(user_input2) !=0 :
        if len(user_input1) ==0 : 
            st.markdown("<span style='color:orange'>Conseil : Vous pouvez rajouter le titre du post pour avoir une prédiction plus fiable</span>",unsafe_allow_html=True)   
        
        st.markdown("<span>Résultat du modèle supervisé :</span>",unsafe_allow_html=True)
        X = TFIDF.transform([text]).toarray()
        X = SVD.transform(X)
#         st.markdown(X)
        ypred = ModelSup.predict(X)
        ypred_name = mlb.inverse_transform(ypred)
#       st.markdown(ypred, unsafe_allow_html=True)
        result = Affichage(ypred_name)
        st.markdown(result, unsafe_allow_html=True)
        

        st.markdown("<span>Résultat du modèle non supervisé en utilisant des tags : </span>",unsafe_allow_html=True)
        text = modificationTokenizer(user_input)
        corpus = id2word.doc2bow(text)
        row = lda_model.get_document_topics(corpus)
        row = sorted(row, key=lambda x: x[1], reverse=True)
        main_topics = pd.DataFrame()
        main_topics = pd.DataFrame(row)
        main_topics.columns = ['Num_Topics', 'Proba']
        main_topics['Proba_Score'] = main_topics['Proba']/main_topics['Proba'].sum()*100
        main_topics['Proba_Cum_Score'] = main_topics['Proba_Score'].cumsum()
        N_Topics = main_topics[main_topics['Proba_Score']>8]['Num_Topics']

        DF_tags = pd.DataFrame()
        for i in N_Topics :

            Dict = eval(df_tags_topics[df_tags_topics['Dominant_Topic']==i]['Count_tags'].iloc[0][8:-1])
            Tot = sum(Dict.values())
            DF_tags1 = pd.DataFrame(list(Dict.items()), columns=['Tags', 'Num'])
            DF_tags1['Normalize_Num'] = DF_tags1['Num']/DF_tags1['Num'].sum()
            DF_tags1 = DF_tags1[DF_tags1['Normalize_Num']>0.04]
            DF_tags1['ScoreImportance_Tags'] = (((float(main_topics[main_topics['Num_Topics']==i]['Proba_Score']))/12)**1.5) * DF_tags1['Normalize_Num']
            DF_tags = pd.concat([DF_tags, DF_tags1]) 
            

        df_probTags = DF_tags.groupby(by = ['Tags']).sum()
        L_tags = df_probTags['ScoreImportance_Tags'].sort_values(ascending = False)[df_probTags['ScoreImportance_Tags'].sort_values(ascending = False)>0.4][:5].keys()
        
        result = ''
        for i in L_tags :
            result+= str(i)+ '  '

        st.markdown(result, unsafe_allow_html=True)
        
        st.markdown("<span>Résultat du modèle non supervisé en utilisant des tags : </span>",unsafe_allow_html=True)

    else :
        st.markdown("<span style='color:red'>Analyse impossible : Vous devez entrer au moins le corps du post pour avoir des suggestions de tags</span>",unsafe_allow_html=True)
