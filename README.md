# NLP-Spam-Ham-Classifier &nbsp;&nbsp;[![license](https://img.shields.io/github/license/ajaymache/travis-ci-with-github.svg)](https://opensource.org/licenses/MIT)
A Machine learning classifier to predict whether the SMS is Spam or Ham by using Natural Language Processing(NLP)

### (Natural Language Toolkit)NLTK: 
NLTK is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks.
## Installing NLTK
    !pip install nltk 
Type above code in the Jupyter Notebook or if it doesn’t work in cmd type conda install -c conda-forge nltk. This should work in most cases. 
Install NLTK: http://pypi.python.org/pypi/nltk
### Importing NLTK Library
![1_C03ylTfz2kUwQ55M-V3RxQ](https://user-images.githubusercontent.com/48255425/80208171-449d8880-864d-11ea-82b5-866a882c5f9d.png)

After typing the above, we get an NLTK Downloader Application which is helpful in NLP Tasks

![1_GybEBN089lxn5yaoIy-NEA](https://user-images.githubusercontent.com/48255425/80208294-862e3380-864d-11ea-960c-f30a1cd6ee63.png)

Stopwords Corpus is already installed in my system which helps in removing redundant repeated words. Similarly, we can install other useful packages.
## Reading and Exploring Dataset
### Reading in text data & why do we need to clean the text?
While reading data, we get data in the structured or unstructured format. A structured format has a well-defined pattern whereas unstructured data has no proper structure. In between the 2 structures, we have a semi-structured format which is a comparably better structured than unstructured format.

![1_1wPdgM62H0awdwWaboGpfg](https://user-images.githubusercontent.com/48255425/80208427-cee5ec80-864d-11ea-94e5-9187e3f5a0df.png)


As we can see from above when we read semi-structured data it is hard to interpret so we use pandas to easily understand our data.

![1_lkzwmdKro46F8meMPScgWA](https://user-images.githubusercontent.com/48255425/80208484-f0df6f00-864d-11ea-9452-25d95d735b9f.png)

## Pre-processing Data
Cleaning up the text data is necessary to highlight attributes that we’re going to want our machine learning system to pick up on. Cleaning (or pre-processing) the data typically consists of a number of steps:
### 1. Remove punctuation
Punctuation can provide grammatical context to a sentence which supports our understanding. But for our vectorizer which counts the number of words and not the context, it does not add value, so we remove all special characters. eg: How are you?->How are you

![1_Zaz2s6724l-hqw8du2bkvA](https://user-images.githubusercontent.com/48255425/80208620-3c921880-864e-11ea-8a75-d32daf8d363e.png)

In body_text_clean, we can see that all punctuations like I’ve-> I’ve are omitted.
### 2.Tokenization
Tokenizing separates text into units such as sentences or words. It gives structure to previously unstructured text. eg: Plata o Plomo-> ‘Plata’,’o’,’Plomo’.

![1_iMC5JmPBo3DmuF4k4IUdIg](https://user-images.githubusercontent.com/48255425/80208690-60555e80-864e-11ea-8e53-22838a87d77e.png)

In body_text_tokenized, we can see that all words are generated as tokens.
### 3. Remove stopwords
Stopwords are common words that will likely appear in any text. They don’t tell us much about our data so we remove them. eg: silver or lead is fine for me-> silver, lead, fine.

![1_fZgVH4vptZs1xOkcOgWEhA](https://user-images.githubusercontent.com/48255425/80208818-9b579200-864e-11ea-915a-60eb92436770.png)

In body_text_nostop, all unnecessary words like been, for, the are removed.
## Preprocessing Data: Stemming
Stemming helps reduce a word to its stem form. It often makes sense to treat related words in the same way. It removes suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected. eg: Entitling,Entitled->Entitl
Note: Some search engines treat words with the same stem as synonyms.

![1_SL1p-I2l2hp3AejVyP_Fxw](https://user-images.githubusercontent.com/48255425/80208943-c8a44000-864e-11ea-8929-302bc5a13d8b.png)

In body_text_stemmed, words like entry,wkly is stemmed to entri,wkli even though don’t mean anything.
## Preprocessing Data: Lemmatizing
Lemmatizing derives the canonical form (‘lemma’) of a word. i.e the root form. It is better than stemming as it uses a dictionary-based approach i.e a morphological analysis to the root word.eg: Entitling, Entitled->Entitle
In Short, Stemming is typically faster as it simply chops off the end of the word, without understanding the context of the word. Lemmatizing is slower and more accurate as it takes an informed analysis with the context of the word in mind.

![1_R7AVYyXmGMpxN7KREKohTA](https://user-images.githubusercontent.com/48255425/80209019-ec678600-864e-11ea-8b10-3d98e2a2c467.png)

In body_text_stemmed, we can words like chances are lemmatized to chance whereas it is stemmed to chanc.
## Vectorizing Data
Vectorizing is the process of encoding text as integers i.e. numeric form to create feature vectors so that machine learning algorithms can understand our data.

### Vectorizing Data: Bag-Of-Words
Bag of Words (BoW) or CountVectorizer describes the presence of words within the text data. It gives a result of 1 if present in the sentence and 0 if not present. It, therefore, creates a bag of words with a document-matrix count in each text document.

![1_L8IcuayW9XHqmag41VZZ7w](https://user-images.githubusercontent.com/48255425/80209738-38ff9100-8650-11ea-9507-2066b348f51c.png)

BOW is applied on the body_text, so the count of each word is stored in the document matrix. (Check the repo).
### Vectorizing Data: N-Grams
N-grams are simply all combinations of adjacent words or letters of length n that we can find in our source text. Ngrams with n=1 are called unigrams. Similarly, bigrams (n=2), trigrams (n=3) and so on can also be used.

![1_gFvrHjXeTmWxTGgiqy3LBw](https://user-images.githubusercontent.com/48255425/80209762-4583e980-8650-11ea-9c9d-7a921a39e5ef.png)


Unigrams usually don’t contain much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the letter or word is likely to follow the given word. The longer the n-gram (higher n), the more context you have to work with.

![1_UAmPWGBNbAMWPjE0CL92RA](https://user-images.githubusercontent.com/48255425/80210097-c80ca900-8650-11ea-84b0-c727ebde6e11.png)

N-Gram is applied on the body_text, so the count of each group words in a sentence word is stored in the document matrix.(Check the repo).
### Vectorizing Data: TF-IDF
It computes “relative frequency” that a word appears in a document compared to its frequency across all documents. It is more useful than “term frequency” for identifying “important” words in each document (high frequency in that document, low frequency in other documents).
Note: Used for search engine scoring, text summarization, document clustering.
Check my previous post — In the TF-IDF Section, I have elaborated on the working of TF-IDF.

TF-IDF is applied on the body_text, so the relative count of each word in the sentences is stored in the document matrix. (Check the repo).
Note: Vectorizers outputs sparse matrices. Sparse Matrix is a matrix in which most entries are 0. In the interest of efficient storage, a sparse matrix will be stored by only storing the locations of the non-zero elements.
## Feature Engineering: Feature Creation
Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. It is like an art as it requires domain knowledge and it can tough to create features, but it can be fruitful for ML algorithm to predict results as they can be related to the prediction.

![1_xxxKtXfkm11AQiTnrYqD0Q](https://user-images.githubusercontent.com/48255425/80210081-be834100-8650-11ea-8871-91c7fcb112ef.png)

body_len shows the length of words excluding whitespaces in a message body.
punct% shows the percentage of punctuation marks in a message body.
## Check If Features are good or not

![download](https://user-images.githubusercontent.com/48255425/80209390-9e9f4d80-864f-11ea-8b0a-829f482d4a1e.png)

We can clearly see that Spams have a high number of words as compared to Hams. So it’s a good feature to distinguish.

![download (1)](https://user-images.githubusercontent.com/48255425/80209387-9cd58a00-864f-11ea-8bf7-077bfe123a43.png)

Spam has a percentage of punctuations but not that far away from Ham. Surprising as at times spam emails can contain a lot of punctuation marks. But still, it can be identified as a good feature.

## NLP-Spam-Ham Classifier
All the above-discussed sections are combined to build a Spam-Ham Classifier.

Precision: 1.0 / Recall: 0.862 / F1-Score: 0.926 / Accuracy: 98.027%

![download (2)](https://user-images.githubusercontent.com/48255425/80210629-b2e44a00-8651-11ea-8119-7f341ecfdb58.png)

Random Forest gives an accuracy of 98.027%. High-value F1-score and 100% Precision is also obtained from the model. Confusion Matrix tells us that we correctly predicted 955 hams and 138 spams.0 hams were incorrectly identified as spams and 22 spams were incorrectly predicted as hams. Detecting spams as hams are justifiable as compared to hams as spams.

## Classification Reports
### 1. Using Heatmap

![download](https://user-images.githubusercontent.com/48255425/88396630-6e975000-cde0-11ea-8d8f-221fffc5b2bc.png)

### 2. Using YellowBricks and GaussianNB 

![download (1)](https://user-images.githubusercontent.com/48255425/88396628-6ccd8c80-cde0-11ea-8a1e-89ae294d0cfa.png)




