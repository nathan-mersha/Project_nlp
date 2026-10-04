import joblib
import pandas as pan



# so this funciton cleans up the text and removes the unnnecessary stop words
def cleanup_txt(text):
    my_stop_words = set(nltk.corpus.stopwords.words("english")) # i get all english stop words here, like the a and so on
    lem = nltk.stem.WordNetLemmatizer()
    cleaned_wrds = ""
    for word in nltk.tokenize.word_tokenize(text.lower()):
        if word not in my_stop_words:
            if cleaned_wrds != "":
                cleaned_wrds = cleaned_wrds + " "
            cleaned_wrds = cleaned_wrds + lem.lemmatize(word)
    return cleaned_wrds;


data = "data/raw/SMSSpamCollection"


dev train():
    data = pan.read_ssv(data, sep="\t", names=["label", "text"])
    texts = data['text'].map(cleanup_txt)
    label = []
    labels = (data["label"] == "spam").astype(int)