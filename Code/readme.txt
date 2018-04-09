We start with the viz_0.ipynb file.
We loaded the dataset and searched the column names using the file census_income_metadata.txt'.
There are two datasets with the same columns, one with about 200000 rows and the other with about 100000 rows.
We will use the first to train our model, and the second to check the stability of the model.
We will only use the first dataset to plot graphs and study distributions.

In this file, we will draw for each column, a graph or interesting resources for that value.
We start with the last column, and we see that 94% of individuals earn less than $50,000 per year.

For each column, we then plot the distribution and a cross table to see the proportion of those who earn more or less than $50,000 per year.
When a column has missing values, we fill them in as we think is most accurate. If few are missing, we fill them with the majority value. Otherwise we create an unknown variable.

At the end of this notebook, we export the modified datasets to work on a new file: pred_0.ipynb

In order to use the sklearn library, we must transform all our data into integer.
We start by transforming columns with less than 10 unique values inside into variable dummy.
Then we modify the columns with less than 60 values inside. We reallocate them in whole sorted by the ratio of those who earn more than $50,000/year with those who earn less.
We also create age groups, because the age distribution shows us that a certain age group is more likely to earn more than $50,000/year.

Then we select the features of interest, using Pearson's correlation coefficient.
We have displayed the 20 most correlated columns in the notebook.

Finally, we perform a grid search on a Logistic Regression and a Random Forest Classifier to determine the best characters by maximizing the True Positive Rate. Indeed, there are only 6% of people who earn more than $50,000/year, and I think it is this error that we must reduce.
The models seem overfit but with the Random Forest Classifier, we get with 40 features of interest, a score of.95 on the dev set, and 44% accuracy on the True positive Rate.

It is possible to improve the models by doing gridsearch on other parameters, but it takes time to run. In addition, further work can be done on the columns to find correlations between them.