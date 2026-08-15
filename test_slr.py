import pandas as pd
import slr

data = pd.read_csv("data.csv")

x = data["years_experience"].tolist()
y = data["annual_salary_usd"].tolist()

mylr = slr.LinearRegression(x,y,0.001)
mylr.train(10000)
mylr.test()
