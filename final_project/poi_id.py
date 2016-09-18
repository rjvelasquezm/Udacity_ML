#!/usr/bin/python
import os


#==============================================================================
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)
#==============================================================================

import sys
import pickle
import seaborn as sns
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".




features_list = ['poi','salary','bonus','total_payments',
'exercised_stock_options','total_stock_value','from_this_person_to_poi','from_poi_to_this_person','from_poi_prop','to_poi_prop','restricted_stock','director_fees','shared_receipt_with_poi'] # You will need to use more features
#features_list = features_names
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


features_names = []

for key in data_dict["ALLEN PHILLIP K"]:
    if key != "email_address":
        features_names.append(key)

### Task 2: Remove outliers
#Remove TOTAL entry from dictionary

del data_dict["TOTAL"]


df = pd.DataFrame.from_dict(data_dict,'index')
df.drop(df.columns[[2,3,5,7,8,9,11,12,13,14,17,18,19]],axis=1,inplace=True)

#df=df.replace('NaN',float('nan'))
df=df.replace('NaN',0)
#%matplotlib qt
#sns.pairplot(df,hue='poi')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#Create feature that represents proportion of emails to and from poi instead of raw number
for person in data_dict:
    
    if data_dict[person]['from_this_person_to_poi'] != 'NaN' and data_dict[person]['from_messages'] != 'NaN':
        data_dict[person]['from_poi_prop'] = 1.0*data_dict[person]['from_this_person_to_poi']/data_dict[person]['from_messages']
    else:
        data_dict[person]['from_poi_prop'] = 'NaN'
    
    if data_dict[person]['from_poi_to_this_person'] != 'NaN' and data_dict[person]['to_messages'] != 'NaN':
        data_dict[person]['to_poi_prop'] = 1.0*data_dict[person]['from_poi_to_this_person']/data_dict[person]['to_messages']
    else:
         data_dict[person]['to_poi_prop']  = 'NaN'

    





my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Scale Features
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)

df = pd.DataFrame(features)
df.columns = features_list[1:]

#sns.pairplot(df)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
clf = GaussianNB()
clf = RandomForestClassifier()
clf = LogisticRegression()
#clf = SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.grid_search import GridSearchCV

rf = RandomForestClassifier()
log = LogisticRegression()

params = {'criterion':('gini','entropy'),'max_depth':[5,10,20],'min_samples_split':[2,3,4,5]}
#params = {'C':[0.25,0.5,1,2,5,10]}
clf = GridSearchCV(rf,params)

clf.fit(features,labels)
clf.best_params_



# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)