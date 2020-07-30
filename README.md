# Disaster Response Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
###Explanation of Folders and Files in the Repo

1. App: 
      -Templates:
                -go.html: Renders classifier
                -master.html: Renders web-page
      -run.py: App Routes
      
2. Data:
       -DisasterResponse.db: Merged SQLite Database of categories and messages
       -disaster_messages.csv: Disaster Messages CSV file
       -disaster_categories.csv: Disaster Categories CSV file
       -process_data.py: Cleans data and processess ETL pipeline
      
3. Models:
         -classifier.pkl: Pickel file of the trained classifier
         -train_classifier.py: Script to train Random Forest classifier 
