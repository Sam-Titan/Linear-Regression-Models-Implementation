# Linear Regression Implementations

I am going to be putting here my understanding and Implementation of the Linear Regression model here.

Here I have made three files with different Approaches to Linear Regression:

1. Linear Regression using Mathamatical Formulas  
2. Linear Regression using sklearn  
3. Linear Regression on Random Dataset  

##### In 1 and 2, I have used a small dataset of 10 samples with noise and outliers while in the 3, I have used numpy to generate a random dataset.

Below are the results alongside the graphs.

---

## 1. Linear Regression using Mathamatical Formulas

I am using Least Squares Method to calculate the Linear Model relationship between Independent Variable and Dependant Variable.

### Results

```
Slope (Coefficient): 19.0
Intercept: 3.799999999999997

Predicted Scores:
[22.799999999999997, 41.8, 60.8, 79.8, 98.8, 117.8, 136.8, 155.8, 174.8, 193.8]

Comparing with the actual scores:
[18, 42, 58, 77, 95, 123, 135, 162, 175, 198]

Mean Squared Error: 13.959999999999974
Mean Absolute Error: 3.199999999999997
Root Mean Squared Error: 3.7363083384538776
Coefficient of Determination (R^2): 0.9957104359930065
Adjusted R squared Error: 0.9951742404921322
```

The graph for the same can be found in the repo under the name:

**Linear_Regression_Noise_Outliers_Using_Mathamatical_Formulas**
---

## 2. Linear Regression using sklearn

### Results

```
Slope (Coefficient): 19.836363636363636
Intercept: -0.7999999999999972

Predicted Scores:
[ 19.03636364  38.87272727  58.70909091  78.54545455  98.38181818
 118.21818182 138.05454545 157.89090909 177.72727273 197.56363636]

Mean Squared Error: 8.18909090909094
Mean Absolute Error: 2.4909090909090965
Root Mean Squared Error: 2.861658768807165
Coefficient of Determination (R^2): 0.997483694153751
Adjusted R squared Error: 0.99716915592297
```

The graph for the same can be found in the repo under the name:

**Linear_Regression_Noise_Outliers_Graph**

##### Here you can see despite the both the approaches using the same dataset, the output is slightly different. This is because Approach 1 is only using Formulas while Approach uses different implementation details. These can include Optimized Formulas, Different Intercept Handling, and others.

---

## 3. Linear Regression using Random Dataset

### Results

```
Slope (Coefficient): 3.4553132007706204
Intercept: 1.9337854893777546

Predicted values:
[131.34912718 330.4353548  254.86061849 208.78904181  55.84311228
  55.83477803  22.00349268 301.22477256 209.63784902 246.59503801
   9.04637298 337.06801706 289.56879005  75.30359871  64.76000643
  65.30578789 107.05900932 183.25356803 151.18431798 102.56257475
 213.3481239   50.1332833  102.87891155 128.52327683 159.52024918
 273.23667192  70.92733102 179.61788982 206.6315735   17.98385791
 211.85956017  60.85521105  24.41114829 329.80345778 335.58989659
 281.26038833 107.18738327  35.68255998 238.35772638 154.02025769
  44.10181788 173.03291691  13.81609659 316.13246439  91.35037414
 230.85598498 109.63972509 181.63357538 190.83931001  65.80678953]

Mean Squared Error: 329.2284574994752
Mean Absolute Error: 14.612814403669212
Root Mean Squared Error: 18.144653689158005
Coefficient of Determination (R^2): 0.9673825411815328
Adjusted R squared Error: 0.9667030107894814
```

The graph for the same can be found in the repo under the name:

**Linear_Regression_Random_Dataset_Graph**
