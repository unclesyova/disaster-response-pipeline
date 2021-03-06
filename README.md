# Disaster Response Pipeline Project
The project is a web application that allows to sort messages into certain categories with the aim to recognize messages
reporting disasters.

### Motivation  
The motivation for this project is to design an automated pipeline for quick recognition of messages reporting disasters
and being able to report them to corresponding organizations.

### List of files
* app/run.py -- the main application script
* app/templates -- contains web app templates
* data/disaster_messages.csv -- the dataset with messages on which the ML model was trained and assessed
* data/disaster_categories.csv -- the dataset with categories on which the ML model was trained and assessed
* data/process_data.py -- the script that implements ETL pipeline
* data/Classification.db -- sqlite database that contains the results of ETL pipeline
* models/train_classifier.py -- the script which implements ML pipeline 
* models/model.pkl -- trained ML model for categorizing messages

### How to use
- Download the files listed above
- Run run.py

### 3rd party libraries
The datasets are handeled with numpy(1.19.5) and pandas(0.22.0) libraries.  
nltk(3.2.5) is used for natural language processing.    
Various features of scikit-learn(0.24.1) library are used to train and assess ML model.  
The web app is built with Flask(0.12.2) and uses matplotlib.pyplot(2.1.1) for visualizations.  

### Summary of the results  
ETL pipeline and ML pipeline were implemented and ready to process new messages/categories datasets for building a new model.    
Web app with visualizations of dataset and categorizer for new messages was built.  

### License
This code is under APACHE LICENSE 2.0

### Author & Acknowledgements
Author: Zeev Peisakhovitch  
All the data was provided by: [Figure Eight](https://appen.com/)
