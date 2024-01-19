# CS210 Course Project
[Website for this project](https://kj0ric.github.io/CS210-Course-Project/)
# Motivation 
As an old user of Mi Bands and current user of Garmin Fenix 6, I have been always fascinated by how smartwatches can collect biological data simultaneously while one's personal life continues. Thanks to these small devices, there is no need to participate in a lab experiment to see your valuable biometrics in expanse of risking your sensitive information. 

Wearable technology today still requires huge development regarding accurate measurement. As of this project, I am determined to continue to extract data from my personal wearable devices and deduce personal data to acknowledge my biological patterns. 

That is why I chose to work on this project, where I extract and analyze personal sleep and physical activity data with the aim of correlating different features in my dataset. 

# Data Source
![Garmin Logo](/Images/Garmin-logo.png)

Before starting to work on this project, I knew that wearable technology companies love to monetize collected personal data of their customers. Of course, why not? So, Garmin's API was only accessible to their partner companies. I had to find another way to access my OWN personal data!

Scraping Garmin Connect, the companion app for Garmin wearable devices was tricky, which explains the lack of scraping endeavors on the web. It is a dynamic website with rate limitations, IP blocking and authentication protocols that add an extra layer of complexity. Then, everything changed when I encountered Garmin DB. 

I've collected the data required for my project through [GarminDB](https://github.com/tcgoetz/GarminDB), a collection of well-structured python scripts for parsing health data from Garmin Connect. With several terminal commands I downloaded my health data in .db format. Using SQLite, I converted .db files into human-readable CSV format, and began to preprocess my data. 

# Data Analysis
![JupyterPython](/Images/jupyter-python.png)
All analysis scripts that I used is available in [this Jupyter notebook](/_notebooks/analysis/analysis.md)
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
	- Stress levels and sleep quality
 	- Stress levels and physical activity
		- Label encoding to encode categorical stress level feature
  	- Sleep features and sleep quality
    	- Physical activity and sleep quality
  	- Using Pearson's Correlation Coefficient method
     	- Using Decision Tree model
      		- train and test data  	   	 	  	  		

<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/78f6cf94-8941-4a0c-a0a3-18116b5f4d48" width="300">

<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/483c92ea-a383-4eb8-9584-578886ab46ee" width="300">

<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/e6c85c86-86c6-4058-b90b-71982cc8a97e" width="300">
<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/a1818ef8-247d-460b-bde0-fb44450e9581" width="300">


## Hypothesis Testing and Statistical Analysis
In this final stage, I put forward some hypothesis that I inferred from the EDA stage and prove or disprove them using statistical analysis.
- Sleep duration ~ Sleep quality correlation
- Active calorie burned ~ Sleep score correlation
- Step count ~ Sleep start time correlation
- Stress level ~ Sleep quality

Techniques used in this stage:
- Pearson correlation test
- Chi-square test
- Calculating p-value and deducting hypothesis validity

<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/1c203224-2d02-4617-8e5d-b0d4b91d8c14" width="300">
<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/7c14146c-63f9-41fd-9176-1970cd893f1a" width="300">

# What I learned from my OWN data
I had many assumptions, hypothesis taken for granted that are related to my health data and daily health patterns. Most importantly, I learned that it is highly likely that they do not align with what the data says.

Here, are some hypothesis tested in this project, related to my sleep and physical activity data:

## Hypothesis #1: Sleep Duration and Sleep Quality Correlation
Contrary to my initial assumption, the data reveals a significant positive correlation between sleep duration and sleep quality. This correlation, with a coefficient of 0.63 and an impressively low p-value of 7.4e-24, emerges as the strongest association within the project.

## Hypothesis #2: Active Calorie Burn and Sleep Score Correlation
Surprisingly, there exists a negative correlation, supported by a low p-value, indicating that as I burn more calories during the day, my sleep quality tends to deteriorate. This finding challenges my prior belief that physically active days would result in better sleep quality.

## Hypothesis #3: Step Count and Sleep Start Time Correlation
By the statistical analysis performed, the dataset does not provide evidence to support the notion that my daily step count influences an earlier bedtime, contrary to my previously held logical assumption.

## Hypothesis #4: Daily Stress level and Sleep Quality Correlation
By applying chi-square test to the categorical features, I have found that stress level and sleep quality are correlated, thus not independent from each other. Finally, a conclusion aligning with my initial assumption made my day. 

<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/e0ac7893-bc4b-4029-be94-783af1f6a253" width="300">
<img src="https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/4af43ec8-7da6-4061-a99f-11d45f1e2b70" width="300">

# Limitations and Future Work
In this project, I have suffered from the lack of a big dataset. In the upcoming years, as my health data grows, I am planning to update my findings by feeding new and bigger accumulated data into my analysis methods and seek any contradiction or support in my initial findings. 

Also I would like to devise cleaner and more inclusive ways to collect personal data from Garmin's database. The limitation regarding data collection was time consuming and detrimental for the aim of this project.

![image](https://github.com/Kj0ric/CS210-Course-Project/assets/99014503/5d5cdbd2-1a26-45e5-b404-4d8fc1dd2b94)

