#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:48:06 2019

@author: anooppanyam
"""

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import random
import sklearn.metrics


# Construct a subset of team statistics after 1980
teams = pd.read_csv('Teams.csv')


teams = teams[teams['yearID'] >= 1985]
teams = teams[['yearID', 'teamID', 'Rank', 'R', 'RA', 'G', 'W', 'H', 'BB', 'HBP', 'AB', 'SF', 'HR', '2B', '3B']]
codes = teams.teamID.unique()

teams = teams.set_index(['yearID', 'teamID'])


# Construct a payroll database for each corresponding team,year after 1980

salaries = pd.read_csv('Salaries.csv')

payroll_teams = salaries.groupby(['yearID', 'teamID'])['salary'].sum()


# left join payroll on teams dataframe on the teams index
teams = teams.join(payroll_teams)


#Graphing Utility Functions

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x*1e-6)

def getRandomColor():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())

colorDict = {}
for i in codes:
    colorDict.update({i:getRandomColor()})

formatter = FuncFormatter(millions)


def plot_spending_wins(teams, year):    
    teams_year = teams.xs(year)
    fig, ax = plt.subplots()
    for i in teams_year.index:
        ax.scatter(teams_year['salary'][i], teams_year['W'][i], color=colorDict.get(i), s=100)
        ax.annotate(i, (teams_year['salary'][i], teams_year['W'][i]),
                        bbox=dict(boxstyle="round", color="#0099FF"),
                        xytext=(30, -30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", connectionstyle="angle"))
    ax.xaxis.set_major_formatter(formatter) 
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('Salaries', fontsize=20)
    ax.set_ylabel('Number of Wins' , fontsize=20)
    ax.set_title('Salaries - Wins: '+ str(year), fontsize=25, fontweight='bold')
    plt.show()


plot_spending_wins(teams, 2001)

# Add metrics
teams['BA'] = teams['H']/teams['AB']
teams['OBP'] = (teams['H'] + teams['BB'] + teams['HBP'])/(teams['AB'] + teams['BB'] + teams['HBP'] + teams['SF'])
teams['SLG'] = (teams['H'] + teams['2B'] + (2*teams['3B']) + (3*teams['HR'])) / teams['AB']

#First Model
runs_reg_model1 = sm.ols("R~OBP+SLG+BA",teams)
runs_reg1 = runs_reg_model1.fit()
#Second Model
runs_reg_model2 = sm.ols("R~OBP+SLG",teams)
runs_reg2 = runs_reg_model2.fit()
#Third Model
runs_reg_model3 = sm.ols("R~BA",teams)
runs_reg3 = runs_reg_model3.fit()

print(runs_reg1.summary())
print(runs_reg2.summary())
print(runs_reg3.summary())

"""
BA is overrepresented in ability to generate runs for a team. 
OBP and SLG are better estimators 
"""

