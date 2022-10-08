# Boston-house-price-Dataset

TO DO : Project Topic1

1) Write the algorithm for linear regression from scratch.
2) Download the Boston housing price data
3) Perform basic EDA.
4) Train the model using the algorithm prepared in sep 1.
5) Train the model for linear regression by using sklearn library.
6) Compare the performance of the performance.

Libraries used :

1) pandas
2) Matplotlib
3) NumPy
4) Seaborn
5) sklearn


STEPS:

1) Predict housing prices using machine learing algorithm and perform EDA on the data set.
2) Detect outliers in the continuous columns.
3) EDA & Data Visualizations - Univariate analysis.
4) Plot histogram & Box plot.

5) Treating outliers in the continuous columns. 
- In this data we use a method called Winsorization.
- In this method we define a confidence interval of let's say 90% and then replace all the outliers below the 5th percentile with the value at 5th percentile and all the values above 95th percentile with the value at the 95th percentile.
- It is pretty useful when there are negative values and zeros in the features which cannot be treated with log transforms or square roots.

6) Prediction of house Price:
- "Random Forests" are often used for feature selection in a data science workflow. 
- Scaling the feature variables using " MinMaxScaler ".

7) Train the model for linear regression by using sklearn library.

8) Train the model for Random Forest & Support Vector Machine (SVM).

9) R-squared :
R-squared = (TSS-RSS)/TSS

- A higher R-squared value indicates a higher amount of variability being explained by our model and vice-versa.
- If we had a really low RSS value, it would mean that the regression line was very close to the actual points.
- High RSS value, it would mean that the regression line was far away from the actual points.

# Observations:

- We can see that thr R_squared value for Linera regression is the lowest and the Random Forest is the highest.
- It means that Linear Regression gives us better results on test data, when compared to the other 2 models.







