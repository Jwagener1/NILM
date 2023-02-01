#!/usr/bin/env python3.7
import sys
sys.path.append("/home/debian/.local/lib/python3.7/site-packages/")
import ctypes
from ctypes import c_double, c_int, CDLL
import sys
import os
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from collections import deque
import dash
from dash.dependencies import Output, Input
from dash import dcc
from dash import html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import orjson
from datetime import datetime
import random
import plotly
from quantiphy import Quantity
Quantity.set_prefs(prec=3)
import re
import csv

lib_path = 'python_call_%s.so' % (sys.platform)
lib_path = os.getcwd() + '/' +lib_path
print(lib_path)
try:
    basic_function_lib = CDLL(lib_path)
except:
    print('OS %s not recognized' % (sys.platform))

print(basic_function_lib)


python_mem_init = basic_function_lib.mem_init
python_mem_init.restype = ctypes.POINTER(ctypes.c_int)
python_print_val = basic_function_lib.print_val
python_print_val.restype = ctypes.c_int
python_c_sample = basic_function_lib.c_sample
python_c_sample.restype = None

def sample_using_c(samples,cstring_pointer):
    """Call C function to calculate squares"""
    n = int(samples)
    V_c_array = (c_double * n)()
    I_c_array = (c_double * n)()
    python_c_sample(cstring_pointer,c_int(n), V_c_array,I_c_array)
    return np.array(V_c_array),np.array(I_c_array)
    
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
    # if 0 < i < 0.0095:
    #     I = (I *  11.995)
    # elif 0.0095 < i < 0.016:
    #     I = (I * 11.9086608886015)  + 0.00910819122344381
    # elif 0.016 < i < 0.041:
    #     I = (I * 10.3781342314444)  + 0.00711292951636151
    # elif 0.041 < i < 0.1:
    #     I = (I * 16.42)
    # else:
    I = (I * 16.9312169312169)  - 0.0514285714285714
    return I


app = dash.Dash(__name__)
colors = {
    'background': 'black',
    'text': '#7FDBFF'
}
style_ck={'color': colors['text'], 'family': "Courier New, monospace", 'font-size': 20}
app.layout = html.Div(className="app-body",style={'backgroundColor': colors['background']},
    children = [
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='gauge', animate=False),
        dcc.Interval(id='interval',interval=1000,n_intervals = 0),
        dcc.Store(id="clientside-data",data = []),
        ]),
        
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='power', animate=False),
        ]),
        
        html.Div(style={'width': '80%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='power_plot', animate=False),
        ]),
        
        html.Div(style={'width':'10%', 'height':'100%','float':'right','display': 'inline-block'}, children=[
        dcc.Checklist(id='load_check',options = [
        {
            "label": html.Div(['No Load'], style=style_ck),
            "value": "0",
        },
        {
            "label": html.Div(['Hair Dryer Cold Blast'], style=style_ck),
            "value": "1",
        },
        {
            "label": html.Div(['Hair Dryer Setting 1'], style=style_ck),
            "value": "2",
        },
        {
            "label": html.Div(['Hair Dryer Setting 2'], style=style_ck),
            "value": "3",
        },
        {
            "label": html.Div(['2W Led Light'], style=style_ck),
            "value": "4",
        },
        {
            "label": html.Div(['4W Led Light'], style=style_ck),
            "value": "5",
        },
        {
            "label": html.Div(['5W Led Light'], style=style_ck),
            "value": "6",
        },
        {
            "label": html.Div(['20W CFL Light'], style=style_ck),
            "value": "7",
        },
        {
            "label": html.Div(['70W Incandesant Light'], style=style_ck),
            "value": "8",
        },
        {
            "label": html.Div(['70W Incandesant Light'], style=style_ck),
            "value": "9",
        },
        {
            "label": html.Div(['65W Laptop Charger'], style=style_ck),
            "value": "10",
        },
       
        ],
        labelStyle={"display": "block"},
        value=['0']
        )
        ]),
        
        html.Div(style={'width':'10%', 'height':'100%','float':'right','display': 'inline-block'}, children=[
        dcc.Checklist(id='csv_check',options = [
        {
            "label": html.Div(['Idle'], style=style_ck),
            "value": "0",
        },
        {
            "label": html.Div(['Reset'], style=style_ck),
            "value": "1",
        },
        {
            "label": html.Div(['Append'], style=style_ck),
            "value": "2",
        },
        ],
        value = ['0']
        )
        ]),
        
        html.Div(style={'width': '100%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='fft', animate=False),
        ]),
        
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='I_plot', animate=False),
        ]),
        
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='V_plot', animate=False),
        ]),
        
    ]
)


@app.callback(
    Output('clientside-data', 'data'),
    Input('interval', 'n_intervals')
    )
def update_graph_scatter(n):
    V_inst,I_inst = sample_using_c(cycle_bins,cstring_pointer)
    V_inst = V_cal(V_inst)
    I_inst = I_cal(I_inst)
    V_inst = AA_Filter(h_AA,V_inst)
    I_inst = AA_Filter(h_AA,I_inst)
    V_inst = V_inst[taps::] 
    I_inst = I_inst[taps::] 
    V_inst = V_inst - np.mean(V_inst)
    I_inst = I_inst - np.mean(I_inst)
    
    peak_bins = signal.find_peaks(V_inst,distance = N/2)
    
    k = peak_bins[0][1]
    
    V_inst = V_inst[k:k+3*N]
    I_inst = I_inst[k:k+3*N]
    
    vrms = RMS(V_inst)
    V_rms.append(vrms)
    irms = RMS(I_inst)
    I_rms.append(irms )
    
    #Power Calculation
    P_inst = inst_pow_cal(I_inst,V_inst,h_50)
    #Real power calculation
    P_real.append(real_pow_cal(P_inst))
    
    P_apparent.append(apparent_power(irms,vrms))
    P_reactive.append(np.sqrt((P_apparent[-1]**2) - (P_real[-1]**2)))
    power_factor  = PF(P_real[-1],P_apparent[-1])
    
    time.append(datetime.now().strftime("%H:%M:%S"))
    string = time[-1] + ',' + str(V_rms[-1]) + '\n'

    I_norm = np.array(norm(I_inst))
    win = np.hamming(len(I_norm))
    I_norm = I_norm * win
    #fi_n = rfft(I_norm)
    fft_I = rfft(I_norm)
    s_mag = np.abs(fft_I) * 2 / np.sum(win)
    s_dbfs = 20 * np.log10(s_mag) 
    s_dbfs = s_dbfs[0:241]
    
    H  = np.array(s_dbfs[3::6])
    
    
    # List = [3,P_real[-1],P_reactive[-1],H[0],H[1],H[2],H[3],H[4],H[5],H[6],H[7],
    # H[8],H[9],H[10],H[11],H[12],H[13],H[14],H[15],H[16],H[17],H[18],H[19],H[20],H[21],H[22],
    # H[23],H[24],H[25],H[26],H[27],H[28],H[29],H[30],H[31],H[32],H[33],H[34],H[35],H[36],H[37],
    # H[38],H[39]]
    # with open('data_set_1.csv', 'a', encoding='UTF8', newline='') as f_object:
    #     writer_object = csv.writer(f_object)
    #     writer_object.writerow(List)
    #     f_object.close()
    
 
    
    temp = {'Vrms':list(V_rms),'Irms':list(I_rms),'PF':power_factor,
    'P':list(P_real),'S':list(P_apparent),'Q':list(P_reactive),'fft':list(s_dbfs),
        'I':list(I_inst),'V':list(V_inst),'time':list(time)}
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
    [{'type': 'indicator'}, {'type': 'indicator'}]],horizontal_spacing = 0.15,
    vertical_spacing = 0.3
    )
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
@app.callback(Output('power', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name = "power_trace",
        x = data['P'][-120:],
        y = data['Q'],
        mode= 'markers',
        marker=dict(size=10,color='yellow'),
        ))
    fig.update_layout(
        title="Reactive vs Real Power",
        title_x=0.5,
        xaxis_title="Real Power (W)",
        yaxis_title="Reactive Power (VAR)",
        legend_title="Legend Title",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text'])
    )
    return fig
    
@app.callback(Output('fft', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):

    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name = "FFT(I)",
        x = f[1::],
        y = data['fft'][1::],
        mode= 'lines',
        line=dict(color="#03C4A1", width=1.5),
        ))
    fig.add_trace(go.Scattergl(
        name = "Harmonics",
        x = f[3::6],
        y = data['fft'][3::6],
        mode= 'markers',
        marker=dict(size=10,color='yellow'),
        ))
    fig.update_layout(
        title="FFT",
        title_x=0.5,
        xaxis_title="Frequency (Hz)",
        xaxis_range=[9,3952],
        yaxis_title="Mag (dB)",
        yaxis_range=[-80,1],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text']),
        xaxis=dict(tickmode='linear',tick0=50,dtick=100)
    )
    return fig
    
@app.callback(Output('I_plot', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name = "current_trace",
        #x = t[:2*N],
        #y = data['I'][:2*N],
        x = t,
        y = data['I'],
        mode= 'lines',
        line=dict(color="yellow", width=2),
        ))
    fig.update_layout(
        title="Current",
        title_x=0.5,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (A)",
        yaxis_range=[min(data['I']) -0.01,max(data['I'])+0.01],
        legend_title="Legend Title",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text'])
    )
    return fig
@app.callback(Output('V_plot', 'figure'),
        Input('clientside-data', 'data'),
        )
def update_graph_scatter(data):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name = "volt_trace",
        # x = t[:2*N],
        # y = data['V'][:2*N],
        x = t,
        y = data['V'],
        mode= 'lines',
        line=dict(color="#03C4A1", width=3),
        ))
    fig.update_layout(
        title="Voltage",
        title_x=0.5,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (V)",
        yaxis_range=[-350,350],
        legend_title="Legend Title",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text'])
    )
    return fig 
    
@app.callback(Output('power_plot', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):
    

    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        name = "Power_over_time",
        x = data['time'],
        y = data['P'],
        mode= 'lines',
        line=dict(color="#03C4A1", width=1.5),
        ))
    fig.update_layout(
        title="Power vs Time",
        title_x=0.5,
        xaxis_title="Time ",
        yaxis_title="Power (W)",
        yaxis_range=[0,max(data['P'])+0.15],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text']),
        xaxis=dict(tickformat='%H:%M:%S',tick0=(data['time'][0][:6]),dtick=15,
        minor=dict(ticklen=8, tickcolor="white",dtick=1))
        
    ),
    return fig
    
@app.callback(Output('csv_check', 'value'),
        Input('csv_check', 'value'),
        )

def update_graph_scatter(value):
    print(value)
    print(value[0][0])
    
    return value

if __name__ == '__main__':
    cstring_pointer = python_mem_init()
    
    f0 = 50
    fs = 51.2E3
    dt = 1/fs
    N = int(fs/f0)
    cycle_bins = int(5 * N)
    
    t = np.arange(0, 3*N,1)*dt
    f = rfftfreq(len(t),dt)
    f = f[0:241]
    P_real = deque(maxlen=86400)
    time = deque(maxlen=86400)
    P_apparent = deque(maxlen=360)
    P_reactive = deque(maxlen=360)
    V_rms = deque(maxlen=4)
    I_rms = deque(maxlen=2)
    
    
    #Filter design 50Hz Filter
    #Filter design AA Filter
    taps = 150
    f0 = 51
    h_50 = signal.firwin(taps,f0,window='hamming',fs = fs)
    
    taps = 90
    f0 = 4000
    h_AA = signal.firwin(taps,f0,window='hamming',fs = fs)
    
    
    #app.run_server(host='192.168.7.2',port = 61,debug=True,threaded=True)
    app.run_server(host='0.0.0.0',debug=True,port = 60,threaded=False)