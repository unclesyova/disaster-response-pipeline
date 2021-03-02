# Disaster Response Pipeline Project
The project is a web application that allows to sort messages into certain categories with aim to recognize messages
reporting disasters.

### Motivation  
The motivation for this project is to design an automated pipeline for quick recognition of messages reporting disasters
and being able to report them to corresponding organizations.

### List of files
* app/run.py -- the main application script
* app/templates -- contains web app templates
* data/disaster_messages.csv -- the dataset with messages on which the ML model was trained and assesed
* data/disaster_categories.csv -- the dataset with categories on which the ML model was trained and assesed
* data/process_data.py -- the script that implements ETL pipeline
* data/Classification.db -- sqlite database that contains the results of ETL pipeline
* models/train_classifier.py -- the script which implements ML pipeline 
* models/model.pkl -- trained ML model for categorizing messages

### How to use
- Download the files listed above
- Run run.py

### 3rd party libraries
The datasets are handeled with numpy and pandas libraries.
nltk is used for natural language processing.
Various features of scikit learn library are used to train and asses ML model.
The web app is built with flask and uses pyplot for visualisations.

### Summary of the results  
ETL pipeline and ML pipeline were built and ready 

### License
This code is under APACHE LICENSE 2.0

### Author & Acknowledgements
Author: Zeev Peisakhovitch  
All the data were taken from: https://insights.stackoverflow.com/survey
