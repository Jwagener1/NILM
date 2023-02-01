import csv
header = ['Catagory', 'Real_Power', 'Reactive_Power',
'H0'  , 'H1'  , 'H2'  , 'H3'  , 'H4'  , 'H5'  , 'H6'  , 'H7'  , 'H8'  , 'H9'  ,
'H10' , 'H11' , 'H12' , 'H13' , 'H14' , 'H15' , 'H16' , 'H17' , 'H18' , 'H19' ,
'H20' , 'H21' , 'H22' , 'H23' , 'H24' , 'H25' , 'H26' , 'H27' , 'H28' , 'H29' ,
'H30' , 'H31' , 'H32' , 'H33' , 'H34' , 'H35' , 'H36' , 'H37' , 'H38' , 'H39' ,
]

with open('data_set_1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    