# spam detector version 1
An NLP system cereated with python that catagorizes sms messages as spam or legitimate. 
Created for IU Project : NLP cource

Data : SMS collection ( UCI ) 5,572 labeled messages ( 4,457 train and 1,115 test )
For preprocessing : lower case, tokenization, tf idf encoding, stop word 
Model : Multinomial Naive Bayes, trained on 10%, 50% and 100% of the training data ( according to the assignment)
Evaluation : Separate test set ( 20 percent of the data) 
Accuracy : 0.964 
Precision : 1.000
Recall : 0.732
F1 : 0.845
All scores can  be found in reports/metrics.csv

the Diagram can be found in reports/metrics.png

# How to use this ?

```
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet

python spam_detector.py train
python spam_detector.py predict "your sms message here"

```
