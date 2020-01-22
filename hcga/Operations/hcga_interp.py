# coding: utf-8
# This file is part of hcga.
#
# Copyright (C) 2019,
# Asher Mullokandov (a.mullokandov@imperial.ac.uk),
#
#
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
import xgboost
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns


'''
Collection of SHAP plots for interpreting tree based models. 
Author: Asher Mullokandov, a.mullokandov@imperial.ac.uk
'''

'''
Load data, determine number of classes, create training and test sets.

Assumed:
X = data matrix as pandas dataframe
y = targets as pandas dataframe

model = clf.fit(X_train,y_train.values.ravel())
'''

# Number of classes
num_classes=np.unique(y.values.ravel()).size

explainer = shap.TreeExplainer(model)
train_shap_values = explainer.shap_values(X_train)
test_shap_values = explainer.shap_values(X_test)

# Dataframes of SHAP values
df_train_shap = [pd.DataFrame(train_shap_values[i], columns=X_train.columns.values) for i in range(num_classes)]
df_test_shap  = [pd.DataFrame(test_shap_values[i], columns=X_test.columns.values) for i in range(num_classes)]

def custom_bar_ranking_plot(shap_vals, data, max_feats):

    '''
    Function for customizing and saving SHAP summary bar plot. 

    Arguments:
    shap_vals = SHAP values list generated from explainer
    data      = data to explain
    max_feats = number of features to display

    '''
    plt.rcParams.update({'font.size': 14})
    shap.summary_plot(shap_vals, data, plot_type="bar", max_display=max_feats, show=False)
    fig = plt.gcf()
    fig.set_figheight(20)
    fig.set_figwidth(15)
    #ax = plt.gca()
    plt.tight_layout()
    dataname=[ k for k,v in globals().items() if v is data][0]
    plt.title(f'Feature Rankings-All Classes-{dataname}')
    #plt.savefig(f"SHAP_Feature_Ranking_summary_bar_plot_{dataname}.png")


def custom_dot_summary_plot(shap_vals, data, max_feats):
    '''
    Function for customizing and saving SHAP summary dot plot. 

    Arguments:
    shap_vals = SHAP values list generated from explainer
    data      = data to explain
    max_feats = number of features to display

    '''
    for i in range(num_classes):
        print(f'Sample Expanded Feature Summary for Class {i}')
        plt.rcParams.update({'font.size': 14})
        shap.summary_plot(shap_vals[i], data, plot_type='dot',max_display=max_feats,show=False)
        fig = plt.gcf()
        fig.set_figheight(20)
        fig.set_figwidth(15)
        #ax = plt.gca()
        plt.tight_layout()
        dataname=[ k for k,v in globals().items() if v is data][0]
        plt.title(f'Sample Expanded Feature Summary for Class {i}-{dataname}')
        plt.savefig(f"Sample_Expanded_Feature_Summary_Plot_Class_{i}_{dataname}.png")
        plt.close()


def custom_violin_summary_plot(shap_vals, data, max_feats):
    '''
    Function for customizing and saving SHAP violin plot. 

    Arguments:
    shap_vals = SHAP values list generated from explainer
    data      = data to explain
    max_feats = number of features to display
    '''
    
    for i in range(num_classes):
        print(f'Violin Feature Summary for Class {i}')
        plt.rcParams.update({'font.size': 14})
        shap.summary_plot(shap_vals[i], data, plot_type="violin",max_display=max_feats,show=False)
        fig = plt.gcf()
        fig.set_figheight(20)
        fig.set_figwidth(15)
        #ax = plt.gca()
        plt.tight_layout()
        dataname=[ k for k,v in globals().items() if v is data][0]
        plt.title(f'Violin Feature Summary for Class {i}-{dataname}')
        plt.savefig(f"Vioin_Feature_Summary_Plot_Class_{i}_{dataname}.png")
        plt.close()

# Ranked features

ranked_feats_train = [df_train_shap[i].abs().mean().sort_values(ascending=False).index.tolist() for i in range(num_classes)]

ranked_feats_test  = [df_test_shap[i].abs().mean().sort_values(ascending=False).index.tolist() for i in range(num_classes)]


def multiple_interaction_dep_plot(memb_class, main_feat, shap_vals, data , num_interations):
    '''
    Function to produce several SHAP dependence plots, colored with N=num_interations features to assess
    interactions between features.
    
    Arguments:
    memb_class = membership class
    main_feat  = index of main feature to plot from ranked df, ie, 0 corresponds to most important feature
    shap_vals  = shap values to use
    data       = data to explain
    num_interations = number of dependence plots to produce, ie, number of interactions to test
    '''

    feat = ranked_feats_test[memb_class][main_feat]
    # Approximate interactions
    int_approx = shap.approximate_interactions(feat, shap_vals[memb_class], data)

    # N dependence plots 
    N=num_interations
    for i in range(N):
        shap.dependence_plot(feat, test_shap_values[cl], X_test, interaction_index=int_approx[i], show=False)
        plt.tight_layout()
        dataname=[ k for k,v in globals().items() if v is data][0]
        plt.savefig(f"{feat}_Dependence_Plot_Class_{dataname}_Interaction_{i}.png")
        plt.close()

from matplotlib.colors import ListedColormap

# Dependence Plot. Snippets from poduska@domino.
def custom_dependence_plot(feature, interaction, df_data, df_shap_vals, memb_class, focus_x, focus_y):
    '''
    Function for customizing and saving SHAP dependence plot. 

    Arguments:
    feature = feature to plot
    interaction = feature to color points with
    df_data      = data to explain
    df_shap_vals = SHAP values list generated from explainer
    memb_class = membership class
    focus_x = sample to highlight in plot, x
    focus_y = sample to highlight in plot, y
    '''
    #cmap = ListedColormap(sns.color_palette("RdBu", 200))
    cmap = sns.diverging_palette(240,10,sep=1, l=50, as_cmap=True)
    f, ax = plt.subplots(figsize=(12,8))
    ax.xaxis.grid(True, which='major',linestyle='dashed')
    ax.yaxis.grid(True, which='major',linestyle='dashed')
    ax.set_axisbelow(True)
    points = ax.scatter(df_data[feature],df_shap_vals[memb_class][feature],c=df_data[interaction],s=15,cmap=cmap)
    f.colorbar(points).set_label(interaction)
    ax.scatter(focus_x, focus_y, color='green', s=150)
    plt.xlabel(feature)
    plt.ylabel("SHAP value for " + feature)
    plt.tight_layout()
    dataname=[ k for k,v in globals().items() if v is df_data][0]
    plt.savefig(f"{feature}_Dependence_Plot_{dataname}_Class_{memb_class}_Interaction_{interaction}.png")
    plt.show


