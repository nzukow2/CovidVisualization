import pandas as pd
import numpy as np
import matplotlib as plt

#before import, install pandas and numpy modules at terminal
# (windows uses "pip install", mac uses "pip3 --version")


# CSV obtained from: https://github.com/owid/covid-19-data/tree/master/public/data
# Respective website from Our World in Data: https://ourworldindata.org/coronavirus-source-data

class Visualization:
    def __init__(self):
        pass

    def dataLoad(self):
        pandas.read_csv("owid-covid-data.csv")

#def countryViewer





df = pd.read_csv("owid-covid-data.csv")
pd.options.display.max_rows = 10000000
print(df)