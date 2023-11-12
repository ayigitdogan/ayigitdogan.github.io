---
title: Single Machine Scheduling Optimization with PuLP
date: 2022-03-12 14:10:00 +0800
categories: [Engineering, Optimization]
tags: [linear programming, optimization, scheduling, pulp, python]
render_with_liquid: false
---

In this study, the aim is to model single-machine scheduling problem to minimize the number of tardy jobs with two different approaches and write the necessary code to solve the same problem.
- The first approach suggests writing necessary constraints for every starting time-job possibility to compare them in two pairs and ban any conflict.
- The second approach is based on the necessity that at every time point, there can be at most one job in process.

The below code is written to solve for 5 jobs, whose processing times and due dates are as follows:

| Jobs            | 1 | 2 | 3 | 4 | 5 |
|-----------------|---|---|---|---|---|
| Processing time | 2 | 2 | 1 | 3 | 4 |
| Due date        | 3 | 3 | 1 | 8 | 5 |

The code can be used for any number of jobs with custom parameters by modifying the related variables.

For the LP formulations, further interpretations, and detailed outputs, please refer to the [report](/assets/pdf/Report%20-%20Single%20Machine%20Scheduling%20with%20PuLP.pdf) of this project.


```python
# Import libraries

from pulp import *                      # Will be used as the solver
import numpy    as np                   # Will be used for element-wise calculations
import pandas   as pd                   # Will be used for a better representation of the time horizon

```

## Approach 1



```python
# Definition of the problem

model_i = LpProblem(    name = "single-machine-scheduling", 
                        sense = LpMinimize)

```


```python
# Definition of the sets

J = range(1,6)                      # Set of jobs: J = {1, 2, 3, 4, 5}

```


```python
p = list((2, 2, 1, 3, 4))  # Processing times of each job
d = list((3, 3, 1, 8, 5))  # Due dates of each job

T = range(sum(p))          # Job i can start at t = {0, 1, ... (total processing time - 1)}

colnames = []
for i in T:
    colnames.append(f"t = {i} - {i+1}")
    
timeHorizonDf = pd.DataFrame(columns=colnames)

print("pj: ", p)           # Printing processing times
print("dj: ", d)           # Printing due dates

timeHorizonDf              # Viewing the initial state of the time horizon

```

    pj:  [2, 2, 1, 3, 4]
    dj:  [3, 3, 1, 8, 5]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t = 0 - 1</th>
      <th>t = 1 - 2</th>
      <th>t = 2 - 3</th>
      <th>t = 3 - 4</th>
      <th>t = 4 - 5</th>
      <th>t = 5 - 6</th>
      <th>t = 6 - 7</th>
      <th>t = 7 - 8</th>
      <th>t = 8 - 9</th>
      <th>t = 9 - 10</th>
      <th>t = 10 - 11</th>
      <th>t = 11 - 12</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Definition of the decision variables x(i,j), each of which represents whether the job "j" started at the time "t"

x = {   (j, t):    LpVariable(  name="x-{job}-{time}".format(   job=j, 
                                                                time=t  if t>=10 
                                                                        else "0{}".format(t)),      # Standardizing the time indices with two digits
                                cat="Binary")   for j in J for t in T   }

```


```python
# Statement of the objective function: Number of tardy jobs must be minimized, 
# in other words, maximum # of "x(i,j)'s that cause tardiness" should set to be zero

model_i += lpSum( x[(i, j)]     for i in J                                      # For each job,
                                for j in range( d[i-1] - p[i-1] + 1,            # starting time should not exceed the latest index 
                                                                                # in which the completion in its due date is possible:
                                                                                # From "dj - pj + 1" on, the due date can not be satisfied. 
                                                                                # ("-1"s in the "dj" and "pj" indices are used due to
                                                                                # the indexing style of Python, which starts at 0,
                                                                                # and will be used frequently from this cell on.)

                                                len(T) - p[i-1] + 1)   )        # The final starting dates that each job can be started
                                                                                # without expanding the time horizon longer than 12 periods:
                                                                                # "T - pj" 
                                                                                # (+1 is inserted to satisfy the arguments of the range function.)

```


```python
# Definition of the first type constraints,
# which state that every job must start exactly once within the possible period it can be completed
# without exceeding T periods

for j in J:
    model_i += (    lpSum(    [x[j,t]     for t in range( len(T) - p[j-1] + 1)]   ) == 1,
                    f"job {j} must be started")                                             # Naming the constraints

```


```python
# Definition of the second type constraints,
# which states that possible processing periods of each pair of jobs must not overlap

for j1 in J:                                                                        # corresponds to "j" in the implicit formulation
    for t1 in range(len(T)-p[j1-1] + 1):                                            # corresponds to "t" in the implicit formulation
        ppjobj1 = np.array([t1, t1 + p[j1-1]])                                      # Processing period of job i
        for j2 in J:                                                                # corresponds to "j prime" in the implicit formulation
            if j2 == j1:
                continue                                                            # j' E J \ {j}
            for t2 in range(len(T)-p[j2-1] + 1):                                    # corresponds to "t prime" in the implicit formulation
                ppjobj2 = np.array([t2 + p[j2-1], t2])                              # Processing period of job j (indices are reversed for easier computation)
                if np.prod(np.subtract(ppjobj1, ppjobj2)) < 0:                      # Overlap condition
                    model_i += (    x[(j1, t1)] + x[(j2, t2)] <= 1,                 # Adding one constraint to the model for each overlap case
                                    f"overlap case for x{j1},{t1} and x{j2},{t2}")  # Naming the constraints
                                
```

Solving the model, reporting its status and execution time


```python
%%time

model_i.solve()
LpStatus[model_i.status]

```

    Wall time: 68.5 ms
    




    'Optimal'




```python
# Printing the minimum number of tardy jobs (the objective value),
# and the names of the decision variables that take the value "1" to obtain this objective value

print("z* = ", value(model_i.objective))
for v in model_i.variables():
    if v.varValue == 1:
        print(v.name)

```

    z* =  2
    x_1_01
    x_2_06
    x_3_00
    x_4_03
    x_5_08
    


```python
# Final view of the time horizon

solution1 = []
for v in model_i.variables():
    if v.varValue == 1:
        solution1.append(v.name)                                                    # Creating a list of decision variables that take the value "1"

solution1Split = []
for i in range(len(solution1)):
    solution1Split.append(solution1[i].split("_"))                                  # Splitting their names into a two dimensional list

solution1Split = sorted(solution1Split, 
                        key=lambda l:l[2])                                          # Sorting the list

finalTimeHorizonA = []
for i in J:
    for j in range(p[int(solution1Split[i-1][1]) - 1]):
        finalTimeHorizonA.append(solution1Split[i-1][1])                            # Adjustment of the time horizon

FinalTimeHorizonDf = timeHorizonDf.append(  pd.DataFrame(   [finalTimeHorizonA],    # Inserting the time horizon to the final dataframe
                                                            columns=colnames    ), 
                                            ignore_index=True).rename(  index = {0: "Job # (Model 1)"},
                                                                        inplace = False)

FinalTimeHorizonDf                                                                  # Viewing the dataframe

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t = 0 - 1</th>
      <th>t = 1 - 2</th>
      <th>t = 2 - 3</th>
      <th>t = 3 - 4</th>
      <th>t = 4 - 5</th>
      <th>t = 5 - 6</th>
      <th>t = 6 - 7</th>
      <th>t = 7 - 8</th>
      <th>t = 8 - 9</th>
      <th>t = 9 - 10</th>
      <th>t = 10 - 11</th>
      <th>t = 11 - 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Job # (Model 1)</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## Approach 2


```python
# Definition of the problem

model_ii = LpProblem(   name = "single-machine-scheduling", 
                        sense = LpMinimize)
                        
```


```python
# The objective function and the first type constraints are the same as model i.
# Sets and parameters are also the same, and require no additional code.

model_ii += lpSum( x[(j, t)]    for j in J                                                      # The objective function
                                for t in range( d[j-1] - p[j-1] + 1,       
                                                len(T) - p[j-1] + 1    )       )

for j in J:                                                                                     # The first type constraints
    model_ii += (   lpSum(    [x[j,t]     for t in range( len(T) - p[j-1] + 1)]   ) == 1,
                    f"job {j} must be started")                                                 # Naming the constraints

```


```python
# Definition of the second type constraints,
# which ensure that no overlaps exist "at each time period t"

for t in T:                                                                                     # For each time period t,
    model_ii += (   lpSum( x[(j,s)]     for j in J                                              # all job j's
                                        for s in range( max( 0,                                 # case of being processed at that time period
                                                             t-p[j-1]+1),
                                                        t+1)            
                                                                            ) <= 1  )           # must not overlap.       
           
```

Solving the model, reporting its status and execution time


```python
%%time

model_ii.solve()
LpStatus[model_i.status]

```

    Wall time: 46.3 ms
    




    'Optimal'




```python
# Printing the objective value and the starting times of the jobs

print("z* = ", value(model_ii.objective))
for v in model_ii.variables():
    if v.varValue == 1:
        print(v.name)

```

    z* =  2
    x_1_03
    x_2_01
    x_3_00
    x_4_05
    x_5_08
    


```python
# Final view of the time horizon, same adjustments as in part i

solution2 = []
for v in model_ii.variables():
    if v.varValue == 1:
        solution2.append(v.name)

solution2Split = []
for i in range(len(solution2)):
    solution2Split.append(solution2[i].split("_"))   

solution2Split = sorted(solution2Split, key=lambda l:l[2])

finalTimeHorizonB = []
for i in J:
    for j in range(p[int(solution2Split[i-1][1]) - 1]):
        finalTimeHorizonB.append(solution2Split[i-1][1])  

FinalTimeHorizonDf =    FinalTimeHorizonDf.append(pd.DataFrame([finalTimeHorizonB],  # Inserting the time horizon to the final dataframe
                        columns=colnames), 
                        ignore_index=True).rename(      index = {   0: "Job # (Model 1)",
                                                                1: "Job # (Model 2)"},
                                                        inplace = False)

FinalTimeHorizonDf

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t = 0 - 1</th>
      <th>t = 1 - 2</th>
      <th>t = 2 - 3</th>
      <th>t = 3 - 4</th>
      <th>t = 4 - 5</th>
      <th>t = 5 - 6</th>
      <th>t = 6 - 7</th>
      <th>t = 7 - 8</th>
      <th>t = 8 - 9</th>
      <th>t = 9 - 10</th>
      <th>t = 10 - 11</th>
      <th>t = 11 - 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Job # (Model 1)</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Job # (Model 2)</th>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

*IE 203 - Operations Research II*  
*Boğaziçi University - Industrial Engineering Department*  
[GitHub Repository](https://github.com/ayigitdogan/Single-Machine-Scheduling-with-PuLP)
