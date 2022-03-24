# STAT450NLP

Welcome to the repository for Laura Jabr & Christine Zhao's Senior Capstone Project at Rice University!

Here we implement various ML methods for sentiment analysis. This includes Naive Bayes with sklearn's CountVectorizer, SVM with sklearn's CountVectorizer & BERT word embeddings, 
linear classifier with BERT word embeddings, and LSTM with GloVe word embeddings.

As of 2/20 Naive Bayes & SVM with CountVectorizer have been completed.

## Project Setup

` pip install virtualenv `(if you don't already have virtualenv installed)

` virtualenv venv ` to create your new environment (called 'venv' here)

Run ` venv/Scripts/activate ` to enter the virtual environment

` pip install -r requirements.txt ` to install the requirements in the current environment

#### Supplementary Dataset
To run the LSTM code, you will need to download the GloVe pretrained vectors from [here](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip). This is 2GB and will need to be extracted into the main project directory.

## Updating the Project

1. When pulling changes from origin: ` pip install -r requirements.txt `
2. When updating packages: ` pip freeze > requirements.txt `
