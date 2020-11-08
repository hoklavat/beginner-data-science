#19-01-Absenteeism(Preprocessing)
#Predict probability of employee absenteeism based on different reasons.

import pandas as pd
raw_csv_data = pd.read_csv('AbsenteeismData.csv') #secondary data: obtained from third-party. primary data: preprocesed by us.
raw_csv_data

type(raw_csv_data) #data read as DataFrame type.

df = raw_csv_data.copy() #backup data for further processing.
df
pd.options.display.max_columns = None #display all columns
pd.options.display.max_rows = None #display all rows
display(df) #absenteeism time is dependent to others.
df.info() #check if are there any missing values. non-null means no missing.

#DROP ID COLUMN
df = df.drop(['ID'], axis=1) #row:axis=0, column:axis=1
df

#REASON FOR ABSENCE
df['Reason for Absence'] #show specific column.
df['Reason for Absence'].min() #show minimum value of a specific column.
df['Reason for Absence'].max() #show maximum value of a specific column.
df['Reason for Absence'].unique() #don't show duplicate values of a specific column.
pd.unique(df['Reason for Absence']) #show non-repeating values. same as above.
len(df['Reason for Absence'].unique()) #total number of unique values. reasons.
sorted(df['Reason for Absence'].unique()) #show values of a specific column in a sorted order.

#GET DUMMY VARIABLES
reason_columns = pd.get_dummies(df['Reason for Absence']) #reason of absence for each employee is marked by 1.
reason_columns
reason_columns['check'] = reason_columns.sum(axis=1) #add check column. sum of columns for each employee must be 1.
reason_columns #check=0 missing, check=1 single, check>1 multiple reasons
reason_columns['check'].sum(axis=0) #sum of all check values must be total employee count.
reason_columns['check'].unique() #how many different values in check row? must be 1.
reason_columns = reason_columns.drop(['check'], axis=1) #check is completed. drop check column.
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True) #drop 0 column
reason_columns

#GROUP REASONS FOR ABSENCE
df.columns.values #data set column names
reason_columns.columns.values #reason column values
df = df.drop(['Reason for Absence'], axis=1) #avoid multicollinearity. reasons will be added seperately as 4 groups.
df
reason_columns.loc[:, 1:14].max(axis=1) #Group 1: Reasons 1-14
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1) #Group 1: Reasons 1-14, illness
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1) #Group 2: Reasons 15-17, pregnancy
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1) #Group 3: Reasons 18-20, ...
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1) #Group 4: Reasons 22-28, ...
reason_type_2

#CONCATENATE COLUMN VALUES
df
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1) #add reason groups as columns
df
df.columns.values
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns = column_names
df.head()

#REORDER COLUMNS
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours'] 
df = df[column_names_reordered]
df.head()

#CREATE CHECKPOINT
df_reason_mod = df.copy() #create temporary save for work.
df_reason_mod

#DATE
df_reason_mod['Date']
type(df_reason_mod['Date'])
type(df_reason_mod['Date'][0])
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y') #convert string to timestamp format
df_reason_mod['Date']
type(df_reason_mod['Date'])
type(df_reason_mod['Date'][0])
df_reason_mod.info()

#EXTRACT MONTH
df_reason_mod['Date'][0]
df_reason_mod['Date'][0].month
list_months = []
list_months
df_reason_mod.shape

for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)

list_months
len(list_months)
df_reason_mod['Month Value'] = list_months
df_reason_mod.head()

#EXTRACT DAY OF WEEK
df_reason_mod['Date'][699].weekday()
df_reason_mod['Date'][699]

def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
df_reason_mod.head()
df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod
type(df_reason_date_mod['Transportation Expense'][0])
type(df_reason_date_mod['Distance to Work'][0])
type(df_reason_date_mod['Age'][0])
type(df_reason_date_mod['Daily Work Load Average'][0])
type(df_reason_date_mod['Body Mass Index'][0])

#EDUCATION, CHILDREN, PETS
display(df_reason_date_mod)
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1}) #high-school:0, others are 1
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()

#FINAL CHECKPOINT
df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head(10)
df_preprocessed.to_csv('Preprocessed.csv', index=False)