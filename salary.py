import pandas 
import joblib
from sklearn.linear_model import LinearRegression

ds = pandas.read_csv("SalaryData.csv")
#print(ds)

X = ds['YearsExperience'].values.reshape(30,1)
#print(X.shape)

y = ds['Salary']

model = LinearRegression()
model.fit(X,y)

#print(model.coef_)
while True:
    yofe  = float(input("Enter the year of experience: "))
    print("Salary should be : "  ,  model.predict([[yofe]]))

    quit = input("\nTo Exit press q & To continue press enter :")
    if quit == "q":
        break
joblib.dump(model,"salary.pk1")
