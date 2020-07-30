# Disaster Response Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Go to http://0.0.0.0:3001/
    
### Explanation of Folders and Files in the Repo:

1. app: 
      1. templates:
                -go.html: Renders classifier
                -master.html: Renders web-page,
                
      2. run(2).py: App Routes
      
2. data:
       i. DisasterResponse.db: Merged SQLite Database of categories and messages,
       ii. disaster_messages.csv: Disaster Messages CSV file,
       iii. disaster_categories.csv: Disaster Categories CSV file,
       iv. process.py: Cleans data and processess ETL pipeline,
      
3. models:
         i. classifier.zip: Pickel file of the trained classifier,
         ii. train_classifier.py: Script to train Random Forest classifier. 
### Note:
        - you might have to: pip install plotly_express==0.4.0
