#!/usr/bin/env python3.7
import sys
# sys.path.append("/home/debian/.local/lib/python3.7/site-packages/")
import ctypes
from ctypes import c_double, c_int, CDLL
import sys
import os
import numpy as np
from scipy import signal #Used for the FIR Filters cited in report 
from scipy.fft import rfft, rfftfreq #FFT Libary Code cited in report 
from collections import deque
import dash
from dash.dependencies import Output, Input
from dash import dcc
from dash import html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import orjson #Used to spped up web GUI
from datetime import datetime
import plotly
from quantiphy import Quantity
Quantity.set_prefs(prec=3)
import pickle #Used to import the dictionaries that contain the decison trees 
import re
import pandas as pd
import random
import csv
from datetime import datetime
from datetime import timedelta

   
def inst_pow_cal(I,V,h_50):
    P = I * V
    P = signal.lfilter(h_50, 1.0, P)
    return P

#Real power calculation function
def real_pow_cal(instant_power):
    sum_instant_pow = sum(instant_power)
    num_of_samples = len(instant_power)
    real_power = sum_instant_pow / num_of_samples
    return real_power
 
#Apparent power calculation function
def apparent_power(I_rms,V_rms):
    return I_rms*V_rms

#RMS(Root-Mean-Square) calculation
def RMS(input):
    #Square the input 
    square_input = input * input
    #Take the sum of the squares
    sum_square_input = sum(square_input)
    #Take the mean of the sum of the squares
    mean_sum_square_input = sum_square_input/len(input)
    #Perform the squareroot on the mean of the value
    root_mean_square_input = np.sqrt(mean_sum_square_input)
    return root_mean_square_input

#Power factor calculation
def PF(real_power,apparent_power):
    return real_power / apparent_power


def AA_Filter(h,temp):
    temp = signal.lfilter(h, 1.0, temp)
    return temp

#Normalisation function 
def norm(temp):
    return temp / temp.max()

def V_cal(V):
    V = V * 77.31591804 + 0
    return V

def I_cal(I):
    I = I/2
    i = I - np.mean(I)
    i = max(np.abs(i))
    I = (I * 16.9312169312169)  - 0.0514285714285714
    return I
# This function is directly used from https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/
# with very little modifcations made from the orginal it has been cite and credit given in the report 
def clasificar_datos(observacion, arbol):
    question = list(arbol.keys())[0] 
    if observacion[question.split()[0]] <= float(question.split()[1]):
        answer = arbol[question][0]
    else:
      answer = arbol[question][1]
    # If the answer is not a dictionary
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
    return clasificar_datos(observacion, answer)

# This function is not my own work and has be stated in the report.
def checkMajorityElement(arr, N):
   elem = {}
   for i in range(0, N):
      if arr[i] in elem.keys():
         elem [arr[i]] += 1
      else:
         elem [arr[i]] = 1
   for key in elem:
      if elem [key] > (N / 2):
         return key
   return 0 #The return case defults to 0 if there is no consensus 50/50 chance of being correct. :) 

app = dash.Dash(__name__)
colors = {
    'background': 'black',
    'text': '#7FDBFF'
}

card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}



app.layout = html.Div(className="app-body",style={'backgroundColor': colors['background']},
    children = [
        html.Div(style={'width': '34%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='gauge', animate=False),
        dcc.Interval(id='interval',interval=1000,n_intervals = 0),
        dcc.Store(id="clientside-data",data = []),
        ]),
        html.Div(style={'width': '66%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='pie_chart', animate=False),
        ]),
        html.Div(style={'width': '100%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='power_plot', animate=False),
        dcc.Interval(id='int_power',interval=1000,n_intervals = 0),
        ]), 
    ]
)


@app.callback(
    Output('clientside-data', 'data'),
    Input('interval', 'n_intervals')
    )
def update_graph_scatter(n):

    V_rms = [230]
    I_rms = [1]
    power_factor = 0.7
    P_real = [10]

    P_apparent = [10]

    P_reactive = [10]

    t = datetime.now()
    LED_2W = 2 * random.randint(0, 1)
    LED_4W = 4 * random.randint(0, 1)
    Halogen_1 = 70 * random.randint(0, 1)
    Halogen_2 = 70 * random.randint(0, 1)
    CFL = 20 * random.randint(0, 1)
    Toaster = 750 * random.randint(0, 1)

    Fan_speed_1 = random.randint(0, 1)
    Fan_speed_2 = random.randint(0, 1)

    Fan_speed = 15 * (Fan_speed_1 ^ Fan_speed_2)
    if Fan_speed_2 == 1:
        Fan_speed = Fan_speed + 2
    Total = LED_2W + LED_4W + Halogen_1 + Halogen_2 +CFL + Toaster + Fan_speed
    header = [t.strftime("%H:%M:%S"),LED_2W, LED_4W,Halogen_1 + Halogen_2,CFL,Toaster,Fan_speed,Total]
    with open(os.path.dirname(__file__)+"\\Daily_Energy.csv", 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    f.close()

    
    temp = {'Vrms':list(V_rms),'Irms':list(I_rms),'PF':power_factor,
    'P':list(P_real),'S':list(P_apparent),'Q':list(P_reactive)}
    return temp



@app.callback(Output('gauge', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):
    
    I = Quantity(np.mean(np.array(data['Irms'])),'A')
    I = [re.sub('[a-zA-Z\s]', '', I.render(show_units=False)),''.join(filter(str.isalpha, I.render(show_units=True)))]
    
    P = Quantity(data['P'][-1],'W')
    P = [re.sub('[a-zA-Z\s]', '', P.render(show_units=False)),''.join(filter(str.isalpha, P.render(show_units=True)))]

    fig = make_subplots(
    rows=2,
    cols=2,                   
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
    [{'type': 'indicator'}, {'type': 'indicator'}]],horizontal_spacing = 0.25,vertical_spacing = 0.12)
    if P[1] == 'W':
        if (float(P[0])) < 250:
            fig.add_trace(go.Indicator(
                name = "power_trace",
                value=float(P[0]),
                mode="gauge+number",
                title={'text': "Power"},
                number = {'valueformat':'.f','suffix': P[1]},
            
                gauge={'axis': {'range': [None, 250]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 100], 'color': "green"},
                    {'range': [100, 200], 'color': "orange"},
                {'range': [200, 250], 'color': "red"}],}),
                row=1,
                col=1,)
        else:
            fig.add_trace(go.Indicator(
                name = "power_trace",
                value=float(P[0]),
                mode="gauge+number",
                title={'text': "Power"},
                number = {'valueformat':'.f','suffix': P[1]},
                gauge={'axis': {'range': [None, 1000]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 600], 'color': "green"},
                    {'range': [600, 800], 'color': "orange"},
                    {'range': [800, 1000], 'color': "red"}],}),
                row=1,
                col=1,)
    else:
        fig.add_trace(go.Indicator(
            name = "power_trace",
            value=float(P[0]),
            mode="gauge+number",
            title={'text': "Power"},
            number = {'valueformat':'.f','suffix': P[1]},
            gauge={'axis': {'range': [None, 3.6]},
            'bar': {'color': "black"},

            'steps': [
                {'range': [0, 1.590], 'color': "green"},
                {'range': [1.590, 2.090], 'color': "orange"},
                {'range': [2.090, 3.680], 'color': "red"}],}),
            row=1,
            col=1,)
        
    fig.add_trace(go.Indicator(
        name = "pf_trace",
        value=data['PF'],
        mode="gauge+number",
        title={'text': "Power factor, cos(φ)"},
        number = {'valueformat':'.3f'},
        gauge={'axis': {'range': [None, 1.0]},
           'bar': {'color': "black"},
           'steps': [
               {'range': [0.8, 1.0], 'color': "green"},
               {'range': [0.4, 0.8], 'color': "orange"},
               {'range': [0, 0.4], 'color': "red"}],}),
           row=1,
           col=2,)
    fig.add_trace(go.Indicator(
        name = "volt_trace",
        value=np.mean(np.array(data['Vrms'])),
        mode="gauge+number",
        title={'text': 'Volts'},
        number = {'valueformat':'.2f','suffix': 'V'},
        gauge={'axis': {'range': [207, 253]},
           'bar': {'color': "black"},
           'steps': [
               {'range': [207, 210], 'color': "red"},
               {'range': [210, 220], 'color': "orange"},
               {'range': [220, 240], 'color': "green"},
               {'range': [240, 250], 'color': "orange"},
               {'range': [250, 253], 'color': "red"}],}),
           row=2,
           col=1,)
           
    if I[1] == 'mA':
        fig.add_trace(go.Indicator(
            name = "current_trace",
            value=float(I[0]),
            mode="gauge+number",
            title={'text': 'Current',},
            number = {'valueformat':'.f','suffix': I[1]},
            gauge={'axis': {'range': [None, 1000]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 600], 'color': "green"},
                {'range': [600, 800], 'color': "orange"},
                {'range': [800, 1000], 'color': "red"}],}),
            row=2,
            col=2,)
    else:
        fig.add_trace(go.Indicator(
            name = "current_trace",
            value=float(I[0]),
            mode="gauge+number",
            title={'text': 'Amps',},
            number = {'valueformat':'.f','suffix': I[1]},
            gauge={'axis': {'range': [None, 16]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 10], 'color': "green"},
                {'range': [10, 14], 'color': "orange"},
                {'range': [14, 16], 'color': "red"}],}),
            row=2,
            col=2,)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        font=dict(family="Courier New, monospace",size=18)
    )
    
    return fig

@app.callback(Output('power_plot', 'figure'),
        Input('int_power', 'n_intervals'),
        )

def update_graph_scatter(n):
    
    df = pd.read_csv(os.path.dirname(__file__)+"\\Daily_Energy.csv")
    time = pd.to_datetime(df['Time'],errors='coerce', format ='%H:%M:%S')
    print(type(time))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name = "Total",
        x = time,
        y = df['Total'],
        mode= 'lines',
        line=dict(color="#03C4A1", width=1.5),
        )),
    
    
    fig.update_layout(
        title="Power vs Time",
        title_x=0.5,
        yaxis_title="Power (W)",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text']),
        height=590,    
    ),
    t1 = time.iat[-1] - timedelta(minutes=10)
    fig.update_layout(xaxis_range=[t1,time[-1]])
    return fig

@app.callback(Output('pie_chart', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):

    datafile = pd.read_csv(os.path.dirname(__file__)+"\\Daily_Energy.csv")

    labels = ['LED_2W', 'LED_4W','Halogen','CFL','Toaster','Fan']

    val = []
    for i in labels:
        t = np.sum(datafile[i].to_numpy())/3600000
        t = round(t,5)
        val.append(t)


    fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Daily', 'Montly'])
    fig.add_trace(go.Pie(labels=labels, values=val,name="Daily (kWh)"), 1, 1)
    fig.add_trace(go.Pie(labels=labels, values=[16.67, 16.67, 16.67, 16.67, 16.67,16.67], name="Monthly (kWh)"), 1, 2)
    fig.update_layout(title_text='Appliance Energy Consumption',title_x=0.5)
    
    

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        font=dict(family="Courier New, monospace",size=18)
    )
    
    return fig    
if __name__ == '__main__':
    #app.run_server(host='192.168.7.2',port = 61,debug=True,threaded=True)
    app.run_server(host='0.0.0.0',debug=True,port = 60,threaded=False)