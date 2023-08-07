import pandas as pd
# Splitting the data into x and y, train and test sets
from sklearn.model_selection import train_test_split
# picke is very useful for saving a model
# We are just saving model parameters
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')
def get_title(name):
  if '.' in name:
    return name.split(',')[1].split('.')[0].strip()
  else:
    return 'Unknown'


# Normalize the titles
def replace_titles(x):
  title = x['Title']
  if title in ['Capt', 'Col', 'Major']:
    return 'Officer'
  elif (title in ['Jonkheer', 'Don', 'the Countess', 'Dona', 'Lady', 'Sir'  ]):
    return "Royalty"
  elif title == "Mme":
    return 'Mrs'
  elif title in ['Mlle', 'Ms']:
    return "Miss"
  else:
    return title

df['Title'] = df['Name'].map(lambda x: get_title(x) )
df['Title'] = df.apply(replace_titles, axis=1)

# COMPLETING
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
# Droping not useful columns
df.drop('Cabin', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
# CONVERTING
df.Sex.replace(('male', 'female'), (0, 1), inplace=True)
df.Embarked.replace(('S', 'C', 'Q'), (0, 1, 2), inplace=True)
df.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Officer', 'Royalty'), (0, 1, 2, 3, 4, 5, 6, 7), inplace=True)


# Taking the independent variables and droping the ependent variable Survived and a not useful
# variable for machine learning PassangerId
x = df.drop(['Survived', 'PassengerId'], axis=1)
# Taking the depended variable
y = df['Survived']
# Creating the training and validation datasets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)



# Store de model in a variable
randomforest = RandomForestClassifier()
# Training the model
randomforest.fit(x_train, y_train)

y_predict = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_predict, y_val)*100, 2)
print('Accuracy: {}'.format(acc_randomforest))

filename = 'titanic_model.sav'
# wb --> write binary
pickle.dump(randomforest, open(filename, 'wb'))

