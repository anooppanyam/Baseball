#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:11:57 2019

@author: anooppanyam
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn import metrics



# Connect to baseball db
cx = sqlite3.connect('lahman2016.sqlite')

# Querying Database for all seasons where a team played 150 or more games and is still active today. 
query = '''select * from Teams 
inner join TeamsFranchises
on Teams.franchID == TeamsFranchises.franchID
where Teams.G >= 150 and TeamsFranchises.active == 'Y';
'''

# Creating dataframe from query.
Teams = cx.execute(query).fetchall()
teams_df = pd.DataFrame(Teams)

teams_df.columns = ['yearID','lgID','teamID','franchID','divID','Rank','G',
                    'Ghome','W','L','DivWin','WCWin',
                    'LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO',
                    'SB','CS','HBP','SF','RA','ER',
                    'ERA','CG','SHO','SV','IPouts','HA','HRA','BBA',
                    'SOA','E','DP','FP','name','park',
                    'attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro',
                    'franchID','franchName','active','NAassoc']


drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin','WCWin',
             'LgWin','WSWin','SF','name','park','attendance','BPF','PPF',
             'teamIDBR','teamIDlahman45','teamIDretro','franchID',
             'franchName','active','NAassoc']

df = teams_df.drop(drop_cols, axis=1)


# Eliminate columns with null values
df = df.drop(['CS','HBP'], axis=1)
# Filling null values with median
df['SO'] = df['SO'].fillna(df['SO'].median())
df['DP'] = df['DP'].fillna(df['DP'].median())


#Visualize distribution of wins
plt.hist(df['W'])
plt.xlabel('Wins')
plt.title('Win Distr.')
plt.show()


# Create bins for wins
def assign_win_bins(W):
    if W < 50:
        return 1
    if W >= 50 and W <= 69:
        return 2
    if W >= 70 and W <= 89:
        return 3
    if W >= 90 and W <= 109:
        return 4
    if W >= 110:
        return 5
 
df['win_bins'] = df['W'].apply(assign_win_bins)

# Plot Year vs Wins
plt.scatter(df['yearID'], df['W'], c=df['win_bins'])
plt.show()

# Eliminate seasons <1990 
df = df[df['yearID'] > 1900]

# Create runs/yr and games/yr to get runs per game in each year

runs_per_year = {}
games_per_year = {}

for i, row in df.iterrows():
    year = row['yearID']
    runs = row['R']
    games = row['G']
    if year in runs_per_year.keys():
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games

runs_per_game = {}

for k,v in runs_per_year.items():
    runs_per_game[k] = v/games_per_year[k]



#Plot runs per game
lists = sorted(runs_per_game.items())
x, y = zip(*lists)

# Create line plot of Year vs. MLB runs per Game
plt.plot(x, y)
plt.title('MLB Yearly Runs per Game')
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')
plt.show()


def assign_label(year):
    if year < 1920:
        return 1
    elif year >= 1920 and year <= 1941:
        return 2
    elif year >= 1942 and year <= 1945:
        return 3
    elif year >= 1946 and year <= 1962:
        return 4
    elif year >= 1963 and year <= 1976:
        return 5
    elif year >= 1977 and year <= 1992:
        return 6
    elif year >= 1993 and year <= 2009:
        return 7
    elif year >= 2010:
        return 8
        
# Add `year_label` column to `df`    
df['year_label'] = df['yearID'].apply(assign_label)

dummy_df = pd.get_dummies(df['year_label'], prefix='era')

# Concatenate `df` and `dummy_df`
df = pd.concat([df, dummy_df], axis=1)

def assign_mlb_rpg(year):
    return runs_per_game[year]

df['mlb_rpg'] = df['yearID'].apply(assign_mlb_rpg)


def assign_decade(year):
    if year < 1920:
        return 1910
    elif year >= 1920 and year <= 1929:
        return 1920
    elif year >= 1930 and year <= 1939:
        return 1930
    elif year >= 1940 and year <= 1949:
        return 1940
    elif year >= 1950 and year <= 1959:
        return 1950
    elif year >= 1960 and year <= 1969:
        return 1960
    elif year >= 1970 and year <= 1979:
        return 1970
    elif year >= 1980 and year <= 1989:
        return 1980
    elif year >= 1990 and year <= 1999:
        return 1990
    elif year >= 2000 and year <= 2009:
        return 2000
    elif year >= 2010:
        return 2010
    
df['decade_label'] = df['yearID'].apply(assign_decade)
decade_df = pd.get_dummies(df['decade_label'], prefix='decade')
df = pd.concat([df, decade_df], axis=1)

# Drop unnecessary columns
df = df.drop(['yearID','year_label','decade_label'], axis=1)

# Create new features for Runs per Game and Runs Allowed per Game
df['R_per_game'] = df['R'] / df['G']
df['RA_per_game'] = df['RA'] / df['G']


fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.scatter(df['R_per_game'], df['W'], c='blue')
ax1.set_title('Runs per Game vs. Wins')
ax1.set_ylabel('Wins')
ax1.set_xlabel('Runs per Game')

ax2.scatter(df['RA_per_game'], df['W'], c='red')
ax2.set_title('Runs Allowed per Game vs. Wins')
ax2.set_xlabel('Runs Allowed per Game')

plt.show()

# Create feature by unsupervised classification
attributes = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA','CG',
'SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1','era_2','era_3',
'era_4','era_5','era_6','era_7','era_8','decade_1910','decade_1920','decade_1930',
'decade_1940','decade_1950','decade_1960','decade_1970','decade_1980',
'decade_1990','decade_2000','decade_2010','R_per_game','RA_per_game','mlb_rpg']

data_attributes = df[attributes]


# Create silhouette score dictionary to find optimal number of clusters
s_score_dict = {}
for i in range(2,11):
    km = KMeans(n_clusters=i, random_state=1)
    l = km.fit_predict(data_attributes)
    s_s = metrics.silhouette_score(data_attributes, l)
    s_score_dict[i] = [s_s]


kmeans_model = KMeans(n_clusters=6, random_state=1)
distances = kmeans_model.fit_transform(data_attributes)

# Create scatter plot using labels from K-means model as color
labels = kmeans_model.labels_

plt.scatter(distances[:,0], distances[:,1], c=labels)
plt.title('Kmeans Clusters')

plt.show()

# Assign labels based on classification
df['labels'] = labels
attributes.append('labels')

# Create new DataFrame using only variables to be included in models
numeric_cols = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA',
                'CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1',
                'era_2','era_3','era_4','era_5','era_6','era_7','era_8','decade_1910','decade_1920',
                'decade_1930','decade_1940','decade_1950','decade_1960','decade_1970','decade_1980',
                'decade_1990','decade_2000','decade_2010','R_per_game','RA_per_game',
                'mlb_rpg','labels','W']
data = df[numeric_cols]
print(data.head())

# Split data DataFrame into train and test sets
train = data.sample(frac=0.75, random_state=1)
test = data.loc[~data.index.isin(train.index)]

x_train = train[attributes]
y_train = train['W']
x_test = test[attributes]
y_test = test['W']


rrm = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), normalize=True)
rrm.fit(x_train, y_train)

# Use model to predict wins
predictions_rrm = rrm.predict(x_test)
mae_rrm = metrics.mean_absolute_error(y_test, predictions_rrm)
print(mae_rrm)