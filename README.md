# Disaster Response Pipeline Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Requirements](#requirements)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

This project is part of the Data Science Nanodegree Program by Udacity in collaboration with figure Eight. The dataset contains pre-labelled tweet and messages from real life disaster events. The aim is to design a model to categorize massages on all 36 pre-defined categoties that can be sent to the appropriate disaster relief agency.

## Requirements <a name="requirements"></a>

The code should run with no issues using Python versions 3 with the following libraries: 
  - Machine Learning: NumPY, Scipy, Pandas, sklearn
  - Natural Language Process: NLTK
  - SQLite Database: SQLalchemy
  - Model Loading and Saving: Pickle
  - Web App and Data Visualization: Flask, Plotly

## File Descriptions <a name="files"></a>

- **Data**
  - disaster_categories.csv + disaster_messages.csv - *Datasets with all the necessary informations*
  - process_data.py - *Code that reads and cleans the csv files and stores it in a SQL database.*
  - db_disaster_messages.db - *Dataset created after the transformed and cleaned data from the disasters CSV files.*
- Models
  - train_classifer.py - *Code necessary to load data and run the machine learning model.*
  - classifier.pkl - *Pickle file from the train_classifer code.*
- App
  - run.py - *Flask app and the user interface used to predict results and display them.*
  - templates - *Folder containing the html template files*

## Results <a name="results"></a>

This is the expected frontpage from the website:

By inputting a sentence it should be able to see the categorie result:



## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eigth for the data. Also, thank you the StackOverFlow community and Udacity  for the training! Otherwise, feel free to use the code here as you would like! 
