'''
TO RUN THIS PROGRAM YOU MUST INSTALL BOKEH PYTHON PACKAGE AND USE BOKEH SERVER:
"bokeh serve --show ucb_vizualized.py"
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from bokeh.core.properties import field
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (ColumnDataSource, HoverTool, SingleIntervalTicker,
                          Slider, Button, Label, Text, Quad, CategoricalColorMapper)
from bokeh.palettes import Spectral10
from bokeh.plotting import figure
from bokeh.models import Legend

import time

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N=10000 # 10,000 users
d=10 # ten ads
ads_selected=[]

numbers_of_selections = [0]*d  #create a vector of size d with zeros
sums_of_rewards = [0]*d #vector for rewards for each ad
total_reward = 0
ads=['ad_1','ad_2','ad_3','ad_4','ad_5','ad_6','ad_7','ad_8','ad_9','ad_10']
ar_matrix = np.zeros((10000,10))
delta_matrix = np.zeros((10000,10))
ad_chosen = np.zeros(10000)

for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (numbers_of_selections[i]>0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/numbers_of_selections[i])
            upper_bound=average_reward+delta_i
            ar_matrix[n][i] = average_reward
            delta_matrix[n][i] = delta_i
            
        else:
            upper_bound = 1e400 # 10^400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
            ad_chosen[n] = ad

        
        
    
    ads_selected.append(ad)
    numbers_of_selections[ad]+=1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad]+=reward
    total_reward += reward


ad1 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,0],
                    'Delta':delta_matrix[:,0],
                    'Upper_bound':ar_matrix[:,0]+delta_matrix[:,0],
		    'Lower_bound':ar_matrix[:,0]-delta_matrix[:,0],
		    'Right_bound':.5,
		    'Left_bound':1.5,
                    'Name':'ad1',
                   'X_coord':1})
ad2 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,1],
                    'Delta':delta_matrix[:,1],
                    'Upper_bound':ar_matrix[:,1]+delta_matrix[:,1],
		    'Lower_bound':ar_matrix[:,1]-delta_matrix[:,1],
		    'Right_bound':1.5,
		    'Left_bound':2.5,
                    'Name':'ad2',
                   'X_coord':2})
ad3 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,2],
                    'Delta':delta_matrix[:,2],
                    'Upper_bound':ar_matrix[:,2]+delta_matrix[:,2],
		    'Lower_bound':ar_matrix[:,2]-delta_matrix[:,2],
		    'Right_bound':2.5,
		    'Left_bound':3.5,
                    'Name':'ad3',
                   'X_coord':3})
ad4 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,3],
                    'Delta':delta_matrix[:,3],
                    'Upper_bound':ar_matrix[:,3]+delta_matrix[:,3],
		    'Lower_bound':ar_matrix[:,3]-delta_matrix[:,3],
		    'Right_bound':3.5,
		    'Left_bound':4.5,
                    'Name':'ad4',
                   'X_coord':4})
ad5 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,4],
                    'Delta':delta_matrix[:,4],
                    'Upper_bound':ar_matrix[:,4]+delta_matrix[:,4],
		    'Lower_bound':ar_matrix[:,4]-delta_matrix[:,4],
		    'Right_bound':4.5,
		    'Left_bound':5.5,
                    'Name':'ad5',
                   'X_coord':5})
ad6 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,5],
                    'Delta':delta_matrix[:,5],
                    'Upper_bound':ar_matrix[:,5]+delta_matrix[:,5],
		    'Lower_bound':ar_matrix[:,5]-delta_matrix[:,5],
		    'Right_bound':5.5,
		    'Left_bound':6.5,
                    'Name':'ad6',
                   'X_coord':6})
ad7 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,6],
                    'Delta':delta_matrix[:,6],
                    'Upper_bound':ar_matrix[:,6]+delta_matrix[:,6],
		    'Lower_bound':ar_matrix[:,6]-delta_matrix[:,6],
		    'Right_bound':6.5,
		    'Left_bound':7.5,
                    'Name':'ad7',
                   'X_coord':7})
ad8 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,7],
                    'Delta':delta_matrix[:,7],
                    'Upper_bound':ar_matrix[:,7]+delta_matrix[:,7],
		    'Lower_bound':ar_matrix[:,7]-delta_matrix[:,7],
		    'Right_bound':7.5,
		    'Left_bound':8.5,
                    'Name':'ad8',
                   'X_coord':8})
ad9 = pd.DataFrame({'Time':np.arange(0,10000),
                    'Average_Rewards':ar_matrix[:,8],
                    'Delta':delta_matrix[:,8],
                    'Upper_bound':ar_matrix[:,8]+delta_matrix[:,8],
		    'Lower_bound':ar_matrix[:,8]-delta_matrix[:,8],
		    'Right_bound':8.5,
		    'Left_bound':9.5,
                    'Name':'ad9',
                   'X_coord':9})
ad10 = pd.DataFrame({'Time':np.arange(0,10000),
                     'Average_Rewards':ar_matrix[:,9],
                     'Delta':delta_matrix[:,9],
                     'Upper_bound':ar_matrix[:,9]+delta_matrix[:,9],
		    'Lower_bound':ar_matrix[:,9]-delta_matrix[:,9],
		    'Right_bound':9.5,
		    'Left_bound':10.5,
                     'Name':'ad10',
                   'X_coord':10})
df=ad1.append(
    ad2,ignore_index=True).append(
    ad3,ignore_index=True).append(
    ad4,ignore_index=True).append(
    ad5,ignore_index=True).append(
    ad6,ignore_index=True).append(
    ad7,ignore_index=True).append(
    ad8,ignore_index=True).append(
    ad9,ignore_index=True).append(
    ad10,ignore_index=True)

df=df.round({'Upper_bound': 4})




#--------------------------------------
# min to max = 10,000 iterations
times = np.arange(0,10000)
    
data = {}

for iteration in times:
    df_iteration = df[df.Time==iteration]
    data[iteration] = df_iteration.reset_index().to_dict('series')

source = ColumnDataSource(data=data[times[0]])

plot = figure(x_range = (1,10),
              y_range = (-.3,.5),
              title='Upper Bound Confidence Algorithm',
              plot_height=180)

plot.xaxis.axis_label = "Ad Number"
plot.yaxis.axis_label = "Confidence metrics"

label = Label(  x=1,y=1,
              text=str(times[0]), text_font_size='70pt', text_color='#eeeeee')
plot.add_layout(label)
color_mapper = CategoricalColorMapper(palette=Spectral10,
                                      factors=list(df.Name.unique()))


plot.quad(
    top='Upper_bound',
    bottom='Lower_bound',
    left='Left_bound',
    right='Right_bound',
    source=source,
    fill_color={'field': 'Name', 'transform': color_mapper},
    fill_alpha=0.2,
    line_color='#7c7e71',
    line_width=0.5,
    line_alpha=0.5,
    legend=field('Name'),
)
plot.text(
    x='X_coord',
    y='Upper_bound',
    text='Upper_bound',
    source=source,
)

plot.square(
    x='X_coord',
    y='Average_Rewards',
    size=10,
    source=source,
    fill_color={'field': 'Name', 'transform': color_mapper},
    fill_alpha=0.8,
    line_color='#7c7e71',
    line_width=0.5,
    line_alpha=0.5,
    legend=field('Name'),
)


plot.add_tools(HoverTool(tooltips="@Name", show_arrow=False, point_policy='follow_mouse'))

def animate_update():
    time = times[slider.value+1]
    if time > times[-1]:
        time = times[0]
        slider.value = 0
    else:
        slider.value = slider.value+1
        
def slider_update(attrname, old, new):
    time = times[slider.value]
    label.text = str(time)
    source.data = data[time]
    
slider = Slider(start=0, end=9999, value=0, step=1, title="Iteration")
slider.on_change('value', slider_update)

callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 1)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [plot],
    [slider, button],
], sizing_mode='scale_width')

curdoc().add_root(layout)
curdoc().title = "Upper Bounds Confidence Algorithm"
