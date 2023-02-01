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
import pickle
import serial


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
python_c_cal = basic_function_lib.calibrate
python_c_cal.restype = None

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
    I = (I * 16.9312169312169)  - 0.0514285714285714
    return I

def csv_reset(file_name):
    header = ['Catagory', 'Real_Power', 'Reactive_Power',
    'H0'  , 'H1'  , 'H2'  , 'H3'  , 'H4'  , 'H5'  , 'H6'  , 'H7'  , 'H8'  , 'H9'  ,
    'H10' , 'H11' , 'H12' , 'H13' , 'H14' , 'H15' , 'H16' , 'H17' , 'H18' , 'H19' ,
    'H20' , 'H21' , 'H22' , 'H23' , 'H24' , 'H25' , 'H26' , 'H27' , 'H28' , 'H29' ,
    'H30' , 'H31' , 'H32' , 'H33' , 'H34' , 'H35' , 'H36' , 'H37' , 'H38' , 'H39' ,
    ]
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    f.close()



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


def load_id(i,state):
    load = ''
    if i == 0:
        load = load + 'Fan Speed 2'
    elif i == 1:
        load = load + 'Fan Speed 1'
    elif i == 2:
        load = load + 'Toaster'
    elif i == 3:
        load = load + 'LED (2W)'
    elif i == 4:
        load = load + 'LED (4W)'
    elif i == 5:
        load = load + 'CFL (20W)'
    elif i == 6:
        load = load + 'Halogen (70W)'    
    elif i == 7:
        load = load + 'Halogen (70W)'

    if state == 0:
        load = load + ' OFF'
    else:
        load = load + ' ON'
    return load


app = dash.Dash(__name__)
colors = {
    'background': 'black',
    'text': '#7FDBFF'
}

style_ck={'color': colors['text'], 'family': "Courier New, monospace",
'font-size': 15,'display': 'flex', 'align-items': 'centre', 'justify-content': 'centre'}
app.layout = html.Div(className="app-body",style={'backgroundColor': colors['background']},
    children = [
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='gauge', animate=False),
        dcc.Interval(id='interval',interval=1000,n_intervals = 0),
        dcc.Store(id="clientside-data",data = []),
        ]),
        
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='I_plot', animate=False),
        ]),
        
        html.Div(style={'width': '75%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='power_plot', animate=False),
        ]),
        
        html.Div(style={'width':'15%','float':'right','display': 'inline-block'}, children=[
            html.H1(id = 'H1', children = 'Loads:', style = {'textAlign':'left',\
                                            'marginTop':3,'marginBottom':3,'color': colors['text'], 'family': "Courier New, monospace",
                                            'font-size': 20,}),
        dcc.Checklist(id='load_check',options = [
        {
            "label": html.Div(['No Load'], style=style_ck),
            "value": 0,
        },
        {
            "label": html.Div(['70W Incandesant Light'], style=style_ck),
            "value": 2**0,
        },
        {
            "label": html.Div(['70W Incandesant Light'], style=style_ck),
            "value": 2**1,
        },
        {
            "label": html.Div(['20W CFL Light'], style=style_ck),
            "value": 2**2,
        },
        {
            "label": html.Div(['2W Led Light'], style=style_ck),
            "value": 2**3,
        },
        {
            "label": html.Div(['4W Led Light'], style=style_ck),
            "value": 2**4,
        },
        {
            "label": html.Div(['Toaster'], style=style_ck),
            "value": 2**5,
        },
        {
            "label": html.Div(['Fan Speed 1'], style=style_ck),
            "value": 2**6,
        },
        {
            "label": html.Div(['Fan Speed 2'], style=style_ck),
            "value": 2**7,
        },
       
        ],
        labelStyle={"display": "block"},
        value=[0]
        )
        ]),
        
        html.Div(style={'width':'10%','height':'50%','float':'right','display': 'inline-block'}, children=[
            html.H1(id = 'H2', children = 'Auto Sampling:', style = {'textAlign':'left',\
                                            'marginTop':3,'marginBottom':3,'color': colors['text'], 'family': "Courier New, monospace",
                                            'font-size': 20,}),
        dcc.RadioItems(id='csv_check',options = [
        {
            "label": html.Div(['Pause'], style=style_ck),
            "value": "0",
        },
        {
            "label": html.Div(['Running'], style=style_ck),
            "value": "1",
        },
        {
            "label": html.Div(['Start'], style=style_ck),
            "value": "2",
        },
        {
            "label": html.Div(['Finished'], style=style_ck),
            "value": "3",
        },
        ],
        labelStyle={"display": "inline"},
        value = '3'
        )
        ]),
    
        
        html.Div(style={'width': '75%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='fft', animate=False),
        ]),
        
        html.Div(style={'width': '25%', 'display': 'inline-block'}, children=[
        dcc.Graph(id='power', animate=False),
        ]),
        
    ]
)


@app.callback(
    Output('clientside-data', 'data'),
    Input('interval', 'n_intervals'),
    Input('csv_check', 'value'),
    )
def data_update(n,csv_check):
    
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
    
    
    V_zero_crossings = np.where(np.diff(np.signbit(V_inst)))[0]
    
    v_freq = fs/(V_zero_crossings[-1] - V_zero_crossings[-3])
    
    I_zero_crossings = np.where(np.diff(np.signbit(I_inst)))[0]
    
    del_t = (I_zero_crossings[-1] - V_zero_crossings[-1])/fs
    
    power_factor_angle = 2*np.pi*v_freq*del_t
    
    
    #Power Calculation
    P_inst = inst_pow_cal(I_inst,V_inst,h_50)
    #Real power calculation
    P_real.append(real_pow_cal(P_inst))
    power_factor  = np.abs(np.cos(power_factor_angle))
    
    P_apparent.append(P_real[-1]/power_factor)
    P_reactive.append(np.sqrt((P_apparent[-1]**2) - (P_real[-1]**2)))
    
    
    time.append(datetime.now().strftime("%H:%M:%S"))
    string = time[-1] + ',' + str(V_rms[-1]) + '\n'

    I_norm = np.array(norm(I_inst))
    win = np.hamming(len(I_norm))
    I_norm = I_norm * win
    fft_I = rfft(I_norm)
    s_mag = np.abs(fft_I) * 2 / np.sum(win)
    s_dbfs = 20 * np.log10(s_mag) 
    s_dbfs = s_dbfs[0:241]
    
    List = list([time[-1],V_rms[-1]])
    with open(file_name, 'a', encoding='UTF8', newline='') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerow(List)
                f_object.close()
    
    
    
    
    if csv_check == '0':
        pass
    elif csv_check == '1':
        if (int(sample_counter[-1]) < 2):
            sample_counter.append(int(sample_counter[-1])+1)
        else:
            sample_counter.append(0)
            load_counter.append(int(load_counter[-1])+1)
            if int(load_counter[-1]) == 192:
                csv_check = '3'
    elif csv_check == '2':
        load_counter.append(0)
        sample_counter.append(0)
    elif csv_check == '3':
        load_counter.append(0)
        sample_counter.append(0)

    temp = {'Vrms':list(V_rms),'Irms':list(I_rms),'PF':power_factor,
    'P':list(P_real),'S':list(P_apparent),'Q':list(P_reactive),'fft':list(s_dbfs),
        'I':list(I_inst),'V':list(V_inst),'time':list(time), 'load_count':list(load_counter),
        'sample_count':list(sample_counter),'csv_con':csv_check
    }
    return temp
    
@app.callback(Output('load_check', 'value'),
        Input('clientside-data', 'data'),
        )

def load_control(data):
    temp = format(int(data['load_count'][-1]), "08b")
    value = []
    value.append(0)
    for i in range(len(temp)):
        if temp[i] == '1':
            value.append(2**(7-i))
        else:
            pass
    if ((2 in value) and (1 not in value)):
        value.remove(2)
        value.append(1)
    if ((64 in value) and (128 in value)):
        if value[-1] == 64:
            value.remove(128)
        else:
            value.remove(64)
    arduino.write(format(sum(value), "08b").encode('ascii'))
    return value
    
@app.callback(Output('csv_check', 'value'),
        Input('load_check', 'value'),
        Input('clientside-data', 'data'),
        )

def update_csv(load_control,data):
    
    value = data['csv_con']
    if value == '0':
        value = '0'
    elif value == '1':
        print(load_control,sum(load_control))
        if int(data['sample_count'][-1]) > 2:
            cat = sum(load_control)
            H = data['fft']
            H  = np.array(H[3::6])
            P_real = np.array(data['P'])
            P_reactive = np.array(data['Q'])
            List = [cat,P_real[-1],P_reactive[-1],H[0],H[1],H[2],H[3],H[4],H[5],H[6],
            H[7],H[8],H[9],H[10],H[11],H[12],H[13],H[14],H[15],H[16],H[17],H[18],
            H[19],H[20],H[21],H[22],H[23],H[24],H[25],H[26],H[27],H[28],H[29],H[30],
            H[31],H[32],H[33],H[34],H[35],H[36],H[37],H[38],H[39]]

            Toast_p = clasificar_datos(List, T1)
            CFL_p = clasificar_datos(List, CFL)
            Hal_p = clasificar_datos(List, Hal)
            Hal2_p = clasificar_datos(List, Hal2)

            F1_1_p = clasificar_datos(List, F1_1)
            F1_2_p = clasificar_datos(List, F1_2)
            F1_3_p = clasificar_datos(List, F1_3)

            F1_p = checkMajorityElement([F1_1_p,F1_2_p,F1_3_p],3)

            F2_1_p = clasificar_datos(List, F2_1)
            F2_2_p = clasificar_datos(List, F2_2)
            F2_3_p = clasificar_datos(List, F2_3)

            F2_p = checkMajorityElement([F2_1_p,F2_2_p,F2_3_p],3)

            L4_1_p = clasificar_datos(List, L4_1)
            L4_2_p = clasificar_datos(List, L4_2)
            L4_3_p = clasificar_datos(List, L4_2)

            L4_p = checkMajorityElement([L4_1_p,L4_2_p,L4_3_p],3)

            L2_1_p = clasificar_datos(List, L2_1)
            L2_2_p = clasificar_datos(List, L2_2)
            L2_3_p = clasificar_datos(List, L2_2)

            L2_p = checkMajorityElement([L2_1_p,L2_2_p,L2_3_p],3)


            a = list(np.binary_repr(load_control, width=8))
            a = np.array(a, dtype=np.int32)
            b = [F1_p,F2_p,Toast_p,L2_p,L4_p,CFL_p,Hal2_p,Hal_p]
            b = np.array(b, dtype=np.int32)

            List = [b[0],a[0],b[1],a[1],b[2],a[2],b[3],a[3],b[4],a[4],b[5],a[5],b[6],a[6],b[7],a[7]]
            with open('Q_Test_1.csv', 'a', encoding='UTF8', newline='') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerow(List)
            f_object.close()




            with open(file_name, 'a', encoding='UTF8', newline='') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerow(List)
                f_object.close()
            value = '1'
    elif value == '2':
        csv_reset(file_name_r)
        value = '1'
    elif value == '3':
        value = '3'
    return value
    

    

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
        x = data['P'],
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
        x = f,
        y = data['fft'],
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
        yaxis_range=[min(data['fft'][3::6]) -2,max(data['fft'][3::6]) + 5 ],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text']),
        xaxis=dict(tickmode='linear',tick0=50,dtick=100)
    )
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
    return fig
    
@app.callback(Output('I_plot', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scattergl(
        name = "I",
        x = t,
        y = data['I'],
        mode= 'lines',
        line=dict(color="yellow", width=2),
        ),
        secondary_y=False,),
        
    fig.add_trace(
        go.Scattergl(
        name = "V",
        x = t,
        y = data['V'],
        mode= 'lines',
        line=dict(color="#03C4A1", width=3),
        ),
        secondary_y=True,),
    fig.update_layout(
        title="Live Current and Voltage Signals",
        title_x=0.5,
        xaxis_title="Time (s)",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text'])
    )
    fig.update_yaxes(
        title_text="Current (A)",
        color="yellow",
        nticks=4,
        secondary_y=False)
    fig.update_yaxes(
        title_text="Volatge (V)",
        color="#03C4A1",
        ticks= "outside",
        tickwidth=2,
        tickcolor='crimson',
        ticklen=10,
        showgrid=False,
        secondary_y=True)
    fig.update_layout(showlegend=False)
    return fig


@app.callback(Output('power_plot', 'figure'),
        Input('clientside-data', 'data'),
        )

def update_graph_scatter(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scattergl(
        name = "Real (W)",
        x = data['time'],
        y = data['P'],
        mode= 'lines',
        line=dict(color="green", width=1.5),
        ),
        secondary_y=False,),
    fig.add_trace(
        go.Scattergl(
        name = "Reactive (VAR)",
        x = data['time'],
        y = data['Q'],
        mode= 'lines',
        textposition="bottom left",
        line=dict(color="red", width=1.5),
        ),
        secondary_y=True,),
    fig.update_layout(
        title="Power vs Time",
        title_x=0.5,
        xaxis_title="Time ",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Courier New, monospace",size=18,color=colors['text']),
        xaxis=dict(tickformat='%H:%M:%S',tick0=(data['time'][0][:6]),dtick=60,
        minor=dict(ticklen=8, tickcolor="white",dtick=15))
        
    ),
    fig.update_yaxes(
        title_text="Real (W)",
        color="green",
        secondary_y=False)
    fig.update_yaxes(
        title_text="Reactive (VAR)",
        color="red",
        secondary_y=True)
    fig.update_layout(showlegend=False)
    return fig
    


    

if __name__ == '__main__':
    cstring_pointer = python_mem_init()
    
    f0 = 50
    fs = 50.2E3
    dt = 1/fs
    N = int(fs/f0)
    cycle_bins = int(5 * N)
    
    t = np.arange(0, 3*N,1)*dt
    f = np.array(rfftfreq(len(t),dt))
    P_real = deque(maxlen=120)
    time = deque(maxlen=120)
    load_ID = deque(maxlen=120)
    P_apparent = deque(maxlen=120)
    P_reactive = deque(maxlen=120)
    V_rms = deque(maxlen=4)
    I_rms = deque(maxlen=3)
    
    load_counter = deque(maxlen=1)
    load_counter.append(0)
    
    sample_counter = deque(maxlen=1)
    sample_counter.append(0)


    T1 = pickle.load(open("T1.p", "rb"))    #Toaster only neeeds one tree
    CLF = pickle.load(open("CLF.p", "rb"))  #CFL only neeeds one tree

    Hal = pickle.load(open("H1.p", "rb"))   #Halogen only neeeds one tree
    Hal2 = pickle.load(open("H2.p", "rb"))  #2xHalogen only neeeds one tree

    F1_1 = pickle.load(open("F1_1.p", "rb")) #Three are needed for fan speed 1
    F1_2 = pickle.load(open("F1_2.p", "rb"))
    F1_3 = pickle.load(open("F1_3.p", "rb"))

    F2_1 = pickle.load(open("F2_1.p", "rb")) #Three are needed for fan speed 2
    F2_2 = pickle.load(open("F2_2.p", "rb"))
    F2_3 = pickle.load(open("F3_3.p", "rb"))

    L4_1 = pickle.load(open("L4_1.p", "rb")) #Three are needed for LED 4
    L4_2 = pickle.load(open("L4_2.p", "rb"))
    L4_3 = pickle.load(open("L4_3.p", "rb"))

    L2_1 = pickle.load(open("L2_1.p", "rb")) #Three are needed for LED 2
    L2_2 = pickle.load(open("L2_2.p", "rb"))
    L2_3 = pickle.load(open("L2_3.p", "rb"))
    
    
    #Filter design 50Hz Filter
    #Filter design AA Filter
    taps = 150
    f0 = 51
    h_50 = signal.firwin(taps,f0,window='hamming',fs = fs)
    
    taps = 90
    f0 = 4000
    h_AA = signal.firwin(taps,f0,window='hamming',fs = fs)
    file_name = 'data.csv'
    file_name_r = 'data.csv'
    arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=19200)
    
    
    #app.run_server(host='192.168.7.2',port = 61,debug=False,threaded=False)
    app.run_server(host='0.0.0.0',debug=False,port = 60,threaded=False)
    arduino.close()