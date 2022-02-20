# STAT450NLP

Welcome to the repository for Laura Jabr & Christine Zhao's Senior Capstone Project at Rice University!

Here we implement various ML methods for sentiment analysis. This includes Naive Bayes with sklearn's CountVectorizer, SVM with sklearn's CountVectorizer & BERT word embeddings, 
linear classifier with BERT word embeddings, and LSTM with GloVe word embeddings.

As of 2/20 Naive Bayes & SVM with CountVectorizer have been completed.

## Project Setup

We exclusively use conda environments for this project. To recreate ours, run `conda create --name STAT450NLP --file spec-file.txt`.

Next, run `conda activate STAT450NLP` in the command line (cmd on Windows) to activate the conda environment. If working in Pycharm, use [this link](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html) to connect the existing Conda environment as an interpreter.

Finally, you'll need to install **pytreebank** to use the Stanford Treebank Dataset (under the folder **sst**) functions in Python. With the conda environment activated in either cmd or Pycharm, run `conda install pip` then `pip install pytreebank`. This extra step is because *pytreebank* and others are examples of libraries only offered through pip.
