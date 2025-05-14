
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries such as pandas, matplotlib, and scikit-learn's LinearRegression module.

2. Read the dataset (student_scores.csv) using pandas and extract the independent variable (Hours) and dependent variable (Scores).

3. Visualize the data using a scatter plot to understand the relationship between hours studied and marks scored.

4. Train the linear regression model using LinearRegression().fit(X, y), where X represents the number of hours studied and y represents the corresponding scores.

5. Make predictions using the trained model on the dataset and also predict for new values (e.g., 6.5 hours).

6. Visualize the regression line over the scatter plot to evaluate the model’s performance.

7. Evaluate the model by retrieving the slope, intercept, and predicted values.

8. Display the plot and the prediction results.

## Program :
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Elavarasan M
RegisterNumber:  212224040083
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()
```


```
df.tail()
```


```
X = df.iloc[:,:-1].values
X
```


```
Y = df.iloc[:,1].values
Y
```


```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
```


```
Y_test
```
```
#graph
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```


```
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```


```
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)
```


## Output :
Head Values 

![img-1](https://github.com/user-attachments/assets/a8f33b86-56c5-497c-bc21-d2d82f6a61e8)


Tail Values

![img-2](https://github.com/user-attachments/assets/9f52f569-6e0b-4b42-be8b-f6b8d8b3fda6)


X values

![img-3](https://github.com/user-attachments/assets/6ceb0f0d-a6aa-40f8-9b4e-426d609fdde1)


Actual  Y Values

![img-4](https://github.com/user-attachments/assets/df10aaa1-ac0e-4011-b7f9-5809e7ff2d40)


Predicted Y values

![img-5](https://github.com/user-attachments/assets/4a8c6b24-752a-4f1f-b041-bc0a48c6c5bb)


Tested Y values

![img-6](https://github.com/user-attachments/assets/14b7c3b5-29ce-49c3-8b95-3cd9b1eba4b2)


Training Data Graph

![img-7](https://github.com/user-attachments/assets/18e8c339-52a1-4e7b-b2b0-fae78b8ef622)


Test Data Graph

![img-8](https://github.com/user-attachments/assets/aa7fa8ba-9d7a-4b44-9113-445dfef70ee2)


Regression Performace Metrics

![img-9](https://github.com/user-attachments/assets/7d120b60-7ad5-4ca2-9d8a-6abc3bf9a67c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
