# Disaster Response Pipeline Project

### Disaster Response Pipeline Project

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. Figure Eight has provided data related to messages, categorized into different classifications, that have been received during emergencies/disasters. The purpose of the project is to create a Natural Language Processing tool that categorize messages.

This is a web application written in python to automatically classify text messages into different categories of disaster management. Furthermore, two visualizations from the data set are available as examples. 

Project Components
There are three components in this project.

1. ETL Pipeline

ETL pipeline (process_data.py) loads the messages and categories datasets. Then it merges the two datasets and cleans the data. Finally, he is stored in a SQLite database

2. ML Pipeline

Model training Data was passed through a pipeline and a prediction model is made. The data loads from the SQLite database. The program splits the dataset into training and test sets. Text processing and machine learning pipeline are created. The model is trained and tuned by using GridSearchCV. The final model is exported as a pickle file. 

3. Flask Web App

Web App show model results in real time.


### Requirements:
You need following python packages:

* flask
* plotly
* sqlalchemy
* pandas
* numpy
* sklearn
* nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
