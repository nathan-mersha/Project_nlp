import joblib
import pandas as pan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

random_stat = 42

metrics_path = "reports/metrics.csv"
model_path = "models/model.joblib"
vectorizer_path = "models/vectorizer.joblib"
def train():
    data = pan.read_ssv(data, sep="\t", names=["label", "text"])
    texts = data['text'].map(cleanup_txt)
    label = []
    labels = (data["label"] == "spam").astype(int)
    textsize=0.2 #means keeping 20 percent of it as a test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=testsize, random_state=random_stat, stratify=labels
    )

    vectorizer = TfidfVectorizer().fit(train_texts)
    tests_v = vectorizer.transform(test_texts)

    results = []
    precent_for_test = [0.1, 0.5, 1.0]
    for franction in precent_for_test:
        s = train_texts.sample(frac=franction, random_state=random_stat)
        s_lbl = train_labels.loc[s.index]

        model = MultinomialNB()
        model.fit(vectorizer.transform(s), s_lbl)
        predictions = model.predict(test_vectors)

        row = [
            len(s),
            accuracy_score(test_labels, predictions),
            precision_score(test_labels, predictions),
            recall_score(test_labels, predictions),
            f1_score(test_labels, predictions)
        ]
        results.append(row)
        columns = ["train_size", "accuracy", "precision", "recall", "f1"]
        pan.DataFrame(results, columns=columns).to_csv(metrics_path, index=False)
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

def predict(text):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    vector = vectorizer.transform([cleanup_txt(text)])
    if model.predict(vector)[0] == 1:
        print("this is a spam")
    else
        print('this is nNot a spam')


command = sys.argv[1] if len(sys.argv) > 1 else ""
message = " ".join(sys.argv[2:])
if command == "train":
    train()
elif command == "predict" and message:
    predict(message)
else
    print(
    ''' i dont know this command, there are 2 availble. 
    python spam_detector.py train - use this first to train on a traing set
    python spam_detector.py predict "your sms message here, put what ever you want spam or ham or chicken for all i care"
    ''')