# Motivation 
As an old user of Mi Bands and current user of Garmin Fenix 6, I have been always fascinated by how smartwatches can collect biological data simultaneously while one's personal live continue. There is no need to participate in a lab experiment to see your valuable biometrics without even risking confidentiality. 

Wearable technology today still requires huge development regarding accurate measurement, and I am determined to continue to extract data from my personal wearable devices and deduce personal data to acknowledge my biological patterns. 

That is why I worked on this project, where I extract and analyze personal sleep and physical activity data with the aim of correlating different features of my dataset. 

# Data Source
![Garmin Logo](/Images/Garmin-logo.png)

Before starting to work on this project, I knew that wearable technology companies seek to monetize collected personal data of their customers. So, Garmin's API was only accessible to their partner companies. I had to find another way to access my OWN personal data!

Scraping Garmin Connect, the companion app for Garmin wearable devices was tricky, which explains the lack of scraping endeavour on the web. It is a dynamic website with rate limiting, IP blocking and authentication protocols that add an extra layer of complexity. Then, I encountered Garmin DB. 

I've collected the data required for my project through [GarminDB](https://github.com/tcgoetz/GarminDB), a collection of python scripts for parsing health data from Garmin Connect. Several terminal commands download your health data in .db format. By using SQLite, I converted .db files into human-readable CSV format, and begin to preprocess my data. 

# Data Analysis
All analysis scripts that I used is available in [this Jupyter notebook]()
As any successfull data science project should, this project includes different data analysis stages and techniques to interpret the extracted data. This is a concise walkthrough of the stages and the techniques I've used:
## Data Cleaning and Preprocessing
- Importing the necessary libraries for all of the scripts
- Reading two CSV files via pandas
- Preparing the dataset for analysis
  - Dropping unnecessary columns and rows
  - Renaming column names into more sensible names
  - Merging two CSV files into one dataframe 
  - Handling missing values by filling with mean
  - Outlier detection using IQR method and handling outliers
  - Handling string values by converting into datetime and timedelta objects
## Exploratory Data Analysis
- Feature engineering
	- Deriving new features from existing features such as stress_level, day_of_week
- Displaying global properties of the dataset using .info() and .describe()
- Exploring the distribution of the features
  	- Normalizing the dataset
  	- Distribution of numerical and categorical features
- Exploring different correlations between features such as
	-  Stress levels and sleep quality
 	-  Stress levels and physical activity
  		- Label encoding to encode categorical stress level feature
    -  Sleep features and sleep quality
    -  Physical activity and sleep quality
    	- using Pearson's Correlation Coefficient method
     	- using Decision Tree model
      		- train and test data  	   	 	  	  		
## Hypothesis Testing and Statistical Analysis

