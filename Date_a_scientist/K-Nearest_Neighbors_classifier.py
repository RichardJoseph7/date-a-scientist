import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# we import the time so we can mesure the time in each run
import time
# to  have an appropriate time, we start it right at the beginning
start_time = time.time()

#Saving the csv file in a df
df = pd.read_csv('profiles.csv')

################################################################

#Here we transform the `drinks' and 'drugs' column into numerical data
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}

#We now create a new column for each of them in our dataset
df["drinks_code"] = df.drinks.map(drink_mapping)
df["drugs_code"] = df.drugs.map(drugs_mapping)

#We also create an age bin so we can classify it later
df['age_range'] = pd.cut(df['age'], [0, 18, 24, 30, 39, 49, 59, 69, 120], labels=['0-18', '19-24', '25-30', '31-39', '40-49', '50-59', '60-69', '69-120'])
age_range_map = {"0-18": 0, "19-24": 1, "25-30": 2, "31-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "69-120": 7}
df["age_range_code"] = df.age_range.map(age_range_map)


# Creating our feature data from the old df, then removing the nan responses
feature_data = df.loc[:,['age', 'age_range_code', 'drugs_code', 'drinks_code']]
feature_data.dropna(inplace=True)

################################################################

#Data exploration
# I wanted to look at 4 different plot at the same time.
'''
#print(df.drinks.value_counts())
#print(df.drinks.head())
#print(df.drinks[1])
#print(df.age.head())
#print(df.age[1])
#print(df.age.mean())
#print(df.drugs.value_counts())
#print(df.drugs[1])

#top right
plt.subplot(2, 2, 1)
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")

#top left
plt.subplot(2, 2, 2)
plt.scatter(df.age, df.drinks_code)
plt.xlabel("age")
plt.ylabel("drinks")

#bottom right
plt.subplot(2, 2, 3)
plt.scatter(df.age, df.drugs_code)
plt.xlabel("age")
plt.ylabel("Drugs")

#bottom left
plt.subplot(2, 2, 4)
plt.scatter(df.drinks_code, df.drugs_code)
plt.xlabel("drinks")
plt.ylabel("Drugs")

plt.show()
'''
################################################################

### KNeighbors Classification
# Creating our training set and our test set
train, test = train_test_split(feature_data, train_size = 0.8, test_size = 0.2,random_state=6)
x_train = train[['drugs_code','drinks_code']]
y_train = train[['age_range_code']]
x_test = test[['drugs_code','drinks_code']]
y_test = test[['age_range_code']]

'''
#List of K Guesses
k_list = range(1, 31)
accuracies = []
for k in range(1, 31):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(x_train, y_train)
  accuracies.append(classifier.score(x_test, y_test))

plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Age VS Drugs and drinks KNeighborsClassifier Accuracy")
plt.plot(k_list, accuracies)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
'''
# creating KNeighbors object with the best K
classifier = KNeighborsClassifier(n_neighbors = 6)
#traning our model witn our training data
classifier.fit(x_train, y_train)
#now lets test out our fit against the test data
y_predicted = classifier.predict(x_test)

# this is our accuracy
print('Our classification accuracy is: ', accuracy_score(y_test, y_predicted))
print("--- %s seconds ---" % (time.time() - start_time))
