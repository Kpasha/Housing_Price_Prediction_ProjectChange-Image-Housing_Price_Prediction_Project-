
# coding: utf-8

# # Predicting price in PricePredictionHousingModelEvaluation

# ### Notebook automatically generated from your model

# Model Gradient Boosted Trees, trained on 2022-06-18 10:39:16.

# #### Generated on 2022-06-18 10:44:50.951960

# prediction
# This notebook will reproduce the steps for a REGRESSION on  PricePredictionHousingModelEvaluation.
# The main objective is to predict the variable price

# #### Warning

# The goal of this notebook is to provide an easily readable and explainable code that reproduces the main steps
# of training the model. It is not complete: some of the preprocessing done by the DSS visual machine learning is not
# replicated in this notebook. This notebook will not give the same results and model performance as the DSS visual machine
# learning model.

# Let's start with importing the required libs :

# In[ ]:


import sys
import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import dataiku.core.pandasutils as pdu
from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter


# And tune pandas display options:

# In[ ]:


pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# #### Importing base data

# The first step is to get our machine learning dataset:

# In[ ]:


# We apply the preparation that you defined. You should not modify this.
preparation_steps = [{'type': 'ColumnRenamer', 'params': {'renamings': [{'from': 'error', 'to': 'error_1'}, {'from': 'error_decile', 'to': 'error_decile_1'}, {'from': 'abs_error_decile', 'to': 'abs_error_decile_1'}, {'from': 'relative_error', 'to': 'relative_error_1'}]}, 'metaType': 'PROCESSOR', 'preview': False, 'disabled': False, 'alwaysShowComment': False}, {'type': 'ColumnsSelector', 'params': {'keep': False, 'appliesTo': 'SINGLE_COLUMN', 'columns': ['prediction']}, 'metaType': 'PROCESSOR', 'preview': True, 'disabled': False, 'alwaysShowComment': False}]
preparation_output_schema = {'columns': [{'name': 'price', 'type': 'double'}, {'name': 'lotsize', 'type': 'bigint'}, {'name': 'bedrooms', 'type': 'bigint'}, {'name': 'bathrms', 'type': 'bigint'}, {'name': 'stories', 'type': 'string'}, {'name': 'driveway', 'type': 'boolean'}, {'name': 'recroom', 'type': 'boolean'}, {'name': 'fullbase', 'type': 'boolean'}, {'name': 'gashw', 'type': 'boolean'}, {'name': 'airco', 'type': 'boolean'}, {'name': 'garagepl', 'type': 'bigint'}, {'name': 'prefarea', 'type': 'boolean'}, {'name': 'prediction', 'type': 'string'}, {'name': 'error_1', 'type': 'double'}, {'name': 'error_decile_1', 'type': 'bigint'}, {'name': 'abs_error_decile_1', 'type': 'bigint'}, {'name': 'relative_error_1', 'type': 'double'}], 'userModified': False}

ml_dataset_handle = dataiku.Dataset('PricePredictionHousingModelEvaluation')
ml_dataset_handle.set_preparation_steps(preparation_steps, preparation_output_schema)
get_ipython().magic('time ml_dataset = ml_dataset_handle.get_dataframe(limit = 100000)')

print ('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)


# #### Initial data management

# The preprocessing aims at making the dataset compatible with modeling.
# At the end of this step, we will have a matrix of float numbers, with no missing values.
# We'll use the features and the preprocessing steps defined in Models.
# 
# Let's only keep selected features

# In[ ]:


ml_dataset = ml_dataset[['error_1', 'stories', 'recroom', 'error_decile_1', 'garagepl', 'relative_error_1', 'bedrooms', 'lotsize', 'gashw', 'driveway', 'price', 'bathrms', 'abs_error_decile_1', 'fullbase', 'airco', 'prefarea']]


# Let's first coerce categorical columns into unicode, numerical features into floats.

# In[ ]:


# astype('unicode') does not work as expected

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x,'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)


categorical_features = ['stories', 'recroom', 'gashw', 'driveway', 'fullbase', 'airco', 'prefarea']
numerical_features = ['error_1', 'error_decile_1', 'garagepl', 'relative_error_1', 'bedrooms', 'lotsize', 'bathrms', 'abs_error_decile_1']
text_features = []
from dataiku.doctor.utils import datetime_to_epoch
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')


# We renamed the target variable to a column named target

# In[ ]:


ml_dataset['__target__'] = ml_dataset['price']
del ml_dataset['price']


# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]


# #### Cross-validation strategy

# The dataset needs to be split into 2 new sets, one that will be used for training the model (train set)
# and another that will be used to test its generalization capability (test set)

# This is a simple cross-validation strategy.

# In[ ]:


train, test = pdu.split_train_valid(ml_dataset, prop=0.8)
print ('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
print ('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))


# #### Features preprocessing

# The first thing to do at the features level is to handle the missing values.
# Let's reuse the settings defined in the model

# In[ ]:


drop_rows_when_missing = []
impute_when_missing = [{'feature': 'error_1', 'impute_with': 'MEAN'}, {'feature': 'error_decile_1', 'impute_with': 'MEAN'}, {'feature': 'garagepl', 'impute_with': 'MEAN'}, {'feature': 'relative_error_1', 'impute_with': 'MEAN'}, {'feature': 'bedrooms', 'impute_with': 'MEAN'}, {'feature': 'lotsize', 'impute_with': 'MEAN'}, {'feature': 'bathrms', 'impute_with': 'MEAN'}, {'feature': 'abs_error_decile_1', 'impute_with': 'MEAN'}]

# Features for which we drop rows with missing values"
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print ('Dropped missing records in %s' % feature)

# Features for which we impute missing values"
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print ('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


# We can now handle the categorical features (still using the settings defined in Models):

# Let's dummy-encode the following features.
# A binary column is created for each of the 100 most frequent values.

# In[ ]:


LIMIT_DUMMIES = 100

categorical_to_dummy_encode = ['stories', 'recroom', 'gashw', 'driveway', 'fullbase', 'airco', 'prefarea']

# Only keep the top 100 values
def select_dummy_values(train, features):
    dummy_values = {}
    for feature in categorical_to_dummy_encode:
        values = [
            value
            for (value, _) in Counter(train[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values
    return dummy_values

DUMMY_VALUES = select_dummy_values(train, categorical_to_dummy_encode)

def dummy_encode_dataframe(df):
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        print ('Dummy-encoded feature %s' % feature)

dummy_encode_dataframe(train)

dummy_encode_dataframe(test)


# Let's rescale numerical features

# In[ ]:


rescale_features = {'error_1': 'AVGSTD', 'error_decile_1': 'AVGSTD', 'garagepl': 'AVGSTD', 'relative_error_1': 'AVGSTD', 'bedrooms': 'AVGSTD', 'lotsize': 'AVGSTD', 'bathrms': 'AVGSTD', 'abs_error_decile_1': 'AVGSTD'}
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print ('Feature %s was dropped because it has no variance' % feature_name)
    else:
        print ('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


# #### Modeling

# Before actually creating our model, we need to split the datasets into their features and labels parts:

# In[ ]:


train_X = train.drop('__target__', axis=1)
test_X = test.drop('__target__', axis=1)

train_Y = np.array(train['__target__'])
test_Y = np.array(test['__target__'])


# Now we can finally create our model !

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(
                    random_state = 1337,
                    verbose = 0,
                    n_estimators = 100,
                    learning_rate = 0.1,
                    loss = 'ls',
                    max_depth = 3
                   )


# ... And train it

# In[ ]:


get_ipython().magic('time clf.fit(train_X, train_Y)')


# Build up our result dataset

# In[ ]:


get_ipython().magic('time _predictions = clf.predict(test_X)')
predictions = pd.Series(data=_predictions, index=test_X.index, name='predicted_value')

# Build scored dataset
results_test = test_X.join(predictions, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'price'})


# #### Results

# You can measure the model's accuracy:

# In[ ]:


c =  results_test[['predicted_value', 'price']].corr()
print ('Pearson correlation: %s' % c['predicted_value'][1])


# That's it. It's now up to you to tune your preprocessing, your algo, and your analysis !
# 
