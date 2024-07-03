#IMPORTING LIBRARIES
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#IMPORTING DATASET
try:
    df = pd.read_csv("tested.csv")
except FileNotFoundError:
    print("The file 'tested.csv' was not found.")
except pd.errors.EmptyDataError:
    print("The file 'tested.csv' is empty.")
except pd.errors.ParserError:
    print("The file 'tested.csv' does not appear to be a valid CSV file.")
else:

    print(df.head(10))

#DATA MANIPULATION
print(df['Survived'].value_counts())

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Passenger class')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Count of Survivors by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

print(df.groupby('Sex')[['Survived']].mean())

labelencoder = LabelEncoder()
df['Sex']= labelencoder.fit_transform(df['Sex'])
print(df.head())

print(df['Sex'], df['Survived'])

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Count of Survivors by Encoded Gender')
plt.xlabel('Sex (0 = female, 1 = male)')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

df=df.drop(['Age'], axis=1)

#MODEL TRAINING
X= df[['Pclass', 'Sex']]
Y=df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)
LogisticRegression(random_state=0)

#MODEL PREDICTION
pred = print(log.predict(X_test))
print(Y_test)