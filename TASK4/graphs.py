import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sympy import Integer
from helpers import class_mapping

def DrawSplit(df, filter: str, upper: int = None, limit: int = float('inf'), add_to_title: str = ""):
    """
    Draws two graphs first with class normal second with anomaly. 
    """
    fig, ax = plt.subplots()
    ax.set_title('Graph contains only normal packages\n' + add_to_title)
    if df['class'][0] == 0 or df['class'][0] == 1:
        loc_df = df.loc[df['class'] == class_mapping[b'normal']]
    else:
        loc_df = df.loc[df['class'] == b'normal']
    Draw(loc_df, filter=filter, upper=upper, show=False, limit=limit, color="green", ax=ax)
    fig, ax = plt.subplots()
    ax.set_title('Graph contains only anomaly packages\n' + add_to_title)
    if df['class'][0] == 0 or df['class'][0] == 1:
        loc_df = df.loc[df['class'] == class_mapping[b'anomaly']]
    else:
        loc_df = df.loc[df['class'] == b'anomaly']
    Draw(loc_df, filter=filter, upper=upper,show=False, limit=limit, color="red", ax=ax)
    plt.show()
def Draw(df, filter: str, upper: int = None, show: bool = True, limit: int = float('inf'), color = "blue", ax=None):
    """
    Draws a graph of elements for a given filer.
    """
    if limit != float('inf'):
        df_process = df[filter][:limit]
    else:
        df_process = df[filter]
    df_process = pd.DataFrame({
        'y_axis': [i if i < upper else upper for i in df_process]
                  if upper != None else df_process,
        'x_axis': range(0, len(df_process))
    })
    if ax == None:
        fig, ax = plt.subplots()
    sns.regplot(ax=ax, x=df_process["x_axis"], y=df_process["y_axis"], 
                line_kws={"color":color,"alpha":0.7,"lw":2}, scatter_kws={"color": color})
    ax.set_ylabel(f'KDD Dataset: {filter}' , size = 12)
    ax.set_xlabel(f'Number of elements' , size = 12)
    if show == True:
        plt.show()