from collections import namedtuple
from numpy.random import uniform as U
import pandas as pd
import numpy as np
import io
import requests

url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
s=requests.get(url).content

names = ['age',
           'workclass',
           'fnlwgt',
           'education',
           'education_num',
           'marital_status',
           'occupation',
           'relationship',
           'race',
           'gender',
           'capital_gain',
           'capital_loss',
           'hours_per_week',
           'native_country',
          'income']

usecols = ['age',
           'workclass',
           'education',
           'marital_status',
           'occupation',
           'relationship',
           'race',
           'gender',
           'hours_per_week',
           'native_country',
           'income']

df_census = pd.read_csv(io.StringIO(s.decode('utf-8')),
                        sep=',',
                        skipinitialspace=True,
                        names=names,
                        header=None,
                        usecols=usecols)

df_census = df_census.replace('?', np.nan).dropna()
edu_map = {'Preschool': 'Elementary',
           '1st-4th': 'Elementary',
           '5th-6th': 'Elementary',
           '7th-8th': 'Elementary',
           '9th': 'Middle',
           '10th': 'Middle',
           '11th': 'Middle',
           '12th': 'Middle',
           'Some-college': 'Undergraduate',
           'Bachelors': 'Undergraduate',
           'Assoc-acdm': 'Undergraduate',
           'Assoc-voc': 'Undergraduate',
           'Prof-school': 'Graduate',
           'Masters': 'Graduate',
           'Doctorate': 'Graduate'}
for from_level, to_level in edu_map.items():
    df_census.education.replace(from_level, to_level, inplace=True)



# Convert raw data to processed data
context_cols = [c for c in usecols if c != 'education']
df_data = pd.concat([pd.get_dummies(df_census[context_cols]), df_census['education']], axis=1)

def get_ad_inventory():
    ad_inv_prob = {'Elementary': 0.9,
                   'Middle':  0.7,
                   'HS-grad':  0.7,
                   'Undergraduate':  0.9,
                   'Graduate':  0.8}
    ad_inventory = []
    for level, prob in ad_inv_prob.items():
        if U() < prob:
            ad_inventory.append(level)
    # Make sure there are at least one ad
    if not ad_inventory:
        ad_inventory = get_ad_inventory()
    return ad_inventory

def get_ad_click_probs():
    base_prob = 0.8
    delta = 0.3
    ed_levels = {'Elementary': 1,
                 'Middle':  2,
                 'HS-grad':  3,
                 'Undergraduate':  4,
                 'Graduate':  5}
    ad_click_probs = {l1: {l2: max(0, base_prob - delta * abs(ed_levels[l1]- ed_levels[l2])) for l2 in ed_levels}
                           for l1 in ed_levels}
    return ad_click_probs

def display_ad(ad_click_probs, user, ad):
    prob = ad_click_probs[ad][user['education']]
    click = 1 if U() < prob else 0
    return click

def calc_regret(user, ad_inventory, ad_click_probs, ad_selected):
    this_p = 0
    max_p = 0
    for ad in ad_inventory:
        p = ad_click_probs[ad][user['education']]
        if ad == ad_selected:
            this_p = p
        if p > max_p:
            max_p = p
    regret = max_p - this_p
    return regret

def get_model(n_input, dropout):
    inputs = keras.Input(shape=(n_input,))
    x = Dense(256, activation='relu')(inputs)
    if dropout > 0:
        x = Dropout(dropout)(x, training=True)
    x = Dense(256, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x, training=True)
    phat = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, phat)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.binary_accuracy])
    return model

def update_model(model, X, y):
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[2]))
    y = np.array(y).reshape(-1)
    model.fit(X, y, epochs=10)
    return model






