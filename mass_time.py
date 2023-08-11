import os
from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
    
    


import warnings
warnings.filterwarnings('ignore')



masses = np.arange(15, 105, 5)

data = pd.DataFrame()
times = pd.DataFrame()
data.insert(loc= 0 , column='MASS', value=masses)

file_dir = '/tmp/binary_c_python-amina/notebooks/notebook_individual_systems/'
all_txt_files = os.listdir(file_dir)

for txt in all_txt_files:
    txt_dir = file_dir + txt    
    new_file = Path(txt_dir).read_text()
    new_file = new_file.strip()
    

    new_file = re.sub(
               r"[ ]+", 
               ";", 
               new_file
           )
    
    
    new_file = re.sub(
           r"[╔╚║]", 
           "", 
           new_file
       )
    
    
    new_file_arr = new_file.split('\n')
    new_file_arr = [el.strip(';') for el in new_file_arr]
    new_file = '\n'.join(new_file_arr)
    
    
    TESTDATA = StringIO(new_file)
    df = pd.read_csv(TESTDATA, sep=';', error_bad_lines = False, skiprows=[1])
#     df[["TIME", "M1/M☉", "TYPE"]]

    time = df[df.TYPE == 'OFF_MS'][['TIME']]
#     print(time)
    times = times.append(time, ignore_index = True)
    
times_sort = times.sort_values(by = "TIME", ascending=False,  ignore_index = True)
# print(times_sort)

new = data.join(times_sort)
new
# new.sort_values(['MASS','TIME'], ascending=[True, False])


from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


xdata=new['MASS']
ydata=new['TIME']
popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)

plt.plot(xdata, func(xdata, *popt), 'r--',
         label='exp: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.plot(xdata, ydata, 'o', markersize=3, label='data')

plt.legend()
plt.show()
