import pandas as pd
import numpy as np
#from pandas_ui import *
from pandas_profiling import ProfileReport

# load dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
url = 'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'

df = pd.read_csv(url)
pr = ProfileReport(df)
pr.to_file(output_file='pandas_profiling1.html')
pr