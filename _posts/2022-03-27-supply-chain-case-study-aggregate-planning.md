---
title: Supply Chain Case Study 1 - Aggregate Planning
date: 2023-01-16 14:10:00 +0800
categories: [Optimization]
tags: [linear programming, pulp, python, supply chain]
render_with_liquid: false
---

*Important note: To get the most out of the below case study, [the solution report](/assets/pdf/Report%20-%20Supply%20Chain%20Case%20Study%201%20-%20Aggregate%20Planning.pdf),
which includes LP formulations and interpretations of the results, can be used as a companion.*

## Case Definition

Can Caravan is a renowned caravan manufacturer, who offers a variety of 42 models to its
customers. These 42 models are grouped under two main categories with respect to their
manufacturing requirements, i.e. basic and pro series. For the June 2022-May 2023 period,
the company wishes to develop an aggregate production plan.

The monthly demand forecast for different caravan series for the planning period is given
below.

<table>
<thead>
  <tr>
    <th></th>
    <th>Basic</th>
    <th>Pro</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Jun.22</td>
    <td>28</td>
    <td>14</td>
  </tr>
  <tr>
    <td>Jul.22</td>
    <td>20</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Aug.22</td>
    <td>26</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Sep.22</td>
    <td>24</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Oct.22</td>
    <td>18</td>
    <td>6</td>
  </tr>
  <tr>
    <td>Nov.22</td>
    <td>10</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Dec.22</td>
    <td>22</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Jan.23</td>
    <td>20</td>
    <td>12</td>
  </tr>
  <tr>
    <td>Feb.23</td>
    <td>18</td>
    <td>6</td>
  </tr>
  <tr>
    <td>Mar.23</td>
    <td>32</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Apr.23</td>
    <td>34</td>
    <td>14</td>
  </tr>
  <tr>
    <td>May.23</td>
    <td>36</td>
    <td>6</td>
  </tr>
</tbody>
</table>

Cost of producing a basic series caravan is estimated to be $6250, excluding cost of direct
labour. This figure is $9750 for pro series caravans. Considering the direct labour
requirements, a basic series product demands 380 man.hours for production, whereas a pro
series caravan requires 530 man.hours. Holding cost of a basic caravan is estimated to be $250
per caravan per month, whereas it costs $500 to hold one unit of pro caravan in stock for a
month. At the end of May 2022, the company projects to have 8 units of basic model, and 3
units of pro model caravans in it stocks.

Currently the company employs 86 assembly workers, who work 180 hours per month on
average. Average monthly salary of an assembly worker is $420. Workers can be asked to work
overtime, which is limited by 40 hours per month. The hourly fee for overtime is 50% more
than the regular hourly fee.

Considering the administrative costs and inefficiencies during the orientation period, cost of
employing a new worker is estimated to be $800 per worker. During lay-offs, the company
pays $1400 per worker.

## Base Problem 

Formulate and solve the aggregate production planning problem. To
develop the aggregate plan, you will need to construct and solve a linear programming
problem minimizing the overall cost comprised of production, holding inventory and
workforce related (regular time and overtime) costs. Shortages are not allowed.
Draw a bar chart of monthly production and inventory for the board meeting.
Comment on your plan including the total production, inventory, workforce used, and
related costs.

In your solution to the base problem, investigate the inventory projections for both
models. When is it optimal to carry base version in the inventory, and when is it
optimal to carry the pro version in the inventory? When do you keep both models in
the inventory? Explain.

## Solution (Base Problem)

```python
# Import libraries

from    pulp    import *                # Will be used as the solver
import  numpy               as np                   
import  matplotlib.pyplot   as plt      # Will be used to generate plots                   

```


```python
# Definition of the problem

model = LpProblem(  name    = "aggregate-planning", 
                    sense   = LpMinimize            )
                    
```


```python
# Definition of the sets

T       = range(1,13)                   # Set of months:                    T = {1, 2, 3, 4 ... 12}
T_ext   = range(13)                     # Extended time horizon             T = 0 included
M       = ("b", "p")                    # Set of aggregated product groups: basic and pro

```


```python
# Definition of the parameters

D = (   (28, 14), 
        (20, 10),
        (26, 4),
        (24, 4),
        (18, 6),
        (10, 8),
        (22, 10),
        (20, 12),
        (18, 6),
        (32, 8),
        (34, 14),
        (36, 6)     )           # Monthly demands

wInit   = 86                    # Initial # of workers
h       = 800                   # Hiring cost [$ / worker]
f       = 1400                  # Firing cost [$ / worker]

c       = (250, 500)            # Inventory costs [$ / month]
iInit   = (8, 3)                # Initial inventory

pCost   = (6250, 9750)          # Production costs [$ / unit]

pWf     = (380, 530)            # Production workforce requirements [man*hours / unit]
p       = 180                   # Default productivity of a worker [hours / month]

r       = 420                   # Regular time employee cost [$ / month]
o       = (420/180)*1.5*40      # Overtime employee cost (regular hourly wage * 1.5 * hours) [$ / month]

oLim    = 40                    # Overtime working limit [hours / month]
```


```python
# Definition of the decision variables

X = {   (i, t): LpVariable( name="X-{model}-{month}".format(    model   = M[i],                                     # Amount of produced models per month
                                                                month   = t     if t>=10 
                                                                                else "0{}".format(t)),              # Standardizing the time indices with two digits
                            lowBound=0                                                                 )
                                                                                for i in range(len(M)) for t in T       }


I = {   (i, t): LpVariable( name="I-{model}-{month}".format(    model   = M[i],                                     # Inventory at the end of the month t
                                                                month   =   t   if t>=10 
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for i in range(len(M)) for t in T_ext   }


W = {   (t): LpVariable(    name="W-{}"             .format(                t   if t>=10                            # # of workers      for     t = 0...12
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T_ext                          }


H = {   (t): LpVariable(    name="H-{}"             .format(                t   if t>=10                            # Hired workers     for     t = 1...12
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


F = {   (t): LpVariable(    name="F-{}"             .format(                t   if t>=10                            # Fired workers     for     t = 1...12
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


O = {   (t): LpVariable(    name="O-{}"             .format(                t   if t>=10                            # Overtime amount   for     t = 1...12
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


```


```python
# Statement of the objective function: The total annual costs must be minimized.

model += lpSum(
                [   pCost[i]*X[(i,t)]   +                                       # Production costs,       
                    c[i]*I[(i,t)]                                               # and inventory holding costs
                                            for t in T                          # are summed over each month,
                                            for i in range(len(M)   )  ] +      # and each product group.
                [   r*W[t] + o*O[t]     +                                       # Wages,
                    h*H[t] + f*F[t]                                             # and hiring/firing costs
                                            for t in T                 ]        # are summed over each month.                            
                                                                            )

```


```python
# Definition of the constraints


model += (W[0]      ==  wInit                                                                   # Setting the initial workforce                                   
                                                                                    )


for i in range(len(M)):                                                                         # Setting the initial inventory levels
    model += (I[(i, 0)] ==  iInit[i]
                                                                                    )


for i in range(len(M)):                                                                         # Inventory balance / demand satisfaction
    for t in T:
        model += (  I[(i,t-1)] + X[(i,t)] == I[(i,t)] + D[t-1][i]
                                                                                    )


for t in T:                                                                                     # Workforce balance
    model += (  W[t] == W[t-1] + H[t] - F[t]
                                                                                    )


for t in T:                                                                                     # # of overtime workers cannot exceed the total workforce
    model += (  O[t] <= W[t]
                                                                                    )            


for t in T:                                                                                     # Capacity constraint
    model += (  lpSum(X[(i,t)]*pWf[i] for i in range(len(M))) <= p*W[t] + oLim*O[t]                              
                                                                                    )


```


```python
model
```




    aggregate-planning:
    MINIMIZE
    1400*F_01 + 1400*F_02 + 1400*F_03 + 1400*F_04 + 1400*F_05 + 1400*F_06 + 1400*F_07 + 1400*F_08 + 1400*F_09 + 1400*F_10 + 1400*F_11 + 1400*F_12 + 800*H_01 + 800*H_02 + 800*H_03 + 800*H_04 + 800*H_05 + 800*H_06 + 800*H_07 + 800*H_08 + 800*H_09 + 800*H_10 + 800*H_11 + 800*H_12 + 250*I_b_01 + 250*I_b_02 + 250*I_b_03 + 250*I_b_04 + 250*I_b_05 + 250*I_b_06 + 250*I_b_07 + 250*I_b_08 + 250*I_b_09 + 250*I_b_10 + 250*I_b_11 + 250*I_b_12 + 500*I_p_01 + 500*I_p_02 + 500*I_p_03 + 500*I_p_04 + 500*I_p_05 + 500*I_p_06 + 500*I_p_07 + 500*I_p_08 + 500*I_p_09 + 500*I_p_10 + 500*I_p_11 + 500*I_p_12 + 140.0*O_01 + 140.0*O_02 + 140.0*O_03 + 140.0*O_04 + 140.0*O_05 + 140.0*O_06 + 140.0*O_07 + 140.0*O_08 + 140.0*O_09 + 140.0*O_10 + 140.0*O_11 + 140.0*O_12 + 420*W_01 + 420*W_02 + 420*W_03 + 420*W_04 + 420*W_05 + 420*W_06 + 420*W_07 + 420*W_08 + 420*W_09 + 420*W_10 + 420*W_11 + 420*W_12 + 6250*X_b_01 + 6250*X_b_02 + 6250*X_b_03 + 6250*X_b_04 + 6250*X_b_05 + 6250*X_b_06 + 6250*X_b_07 + 6250*X_b_08 + 6250*X_b_09 + 6250*X_b_10 + 6250*X_b_11 + 6250*X_b_12 + 9750*X_p_01 + 9750*X_p_02 + 9750*X_p_03 + 9750*X_p_04 + 9750*X_p_05 + 9750*X_p_06 + 9750*X_p_07 + 9750*X_p_08 + 9750*X_p_09 + 9750*X_p_10 + 9750*X_p_11 + 9750*X_p_12 + 0.0
    SUBJECT TO
    _C1: W_00 = 86
    
    _C2: I_b_00 = 8
    
    _C3: I_p_00 = 3
    
    _C4: I_b_00 - I_b_01 + X_b_01 = 28
    
    _C5: I_b_01 - I_b_02 + X_b_02 = 20
    
    _C6: I_b_02 - I_b_03 + X_b_03 = 26
    
    _C7: I_b_03 - I_b_04 + X_b_04 = 24
    
    _C8: I_b_04 - I_b_05 + X_b_05 = 18
    
    _C9: I_b_05 - I_b_06 + X_b_06 = 10
    
    _C10: I_b_06 - I_b_07 + X_b_07 = 22
    
    _C11: I_b_07 - I_b_08 + X_b_08 = 20
    
    _C12: I_b_08 - I_b_09 + X_b_09 = 18
    
    _C13: I_b_09 - I_b_10 + X_b_10 = 32
    
    _C14: I_b_10 - I_b_11 + X_b_11 = 34
    
    _C15: I_b_11 - I_b_12 + X_b_12 = 36
    
    _C16: I_p_00 - I_p_01 + X_p_01 = 14
    
    _C17: I_p_01 - I_p_02 + X_p_02 = 10
    
    _C18: I_p_02 - I_p_03 + X_p_03 = 4
    
    _C19: I_p_03 - I_p_04 + X_p_04 = 4
    
    _C20: I_p_04 - I_p_05 + X_p_05 = 6
    
    _C21: I_p_05 - I_p_06 + X_p_06 = 8
    
    _C22: I_p_06 - I_p_07 + X_p_07 = 10
    
    _C23: I_p_07 - I_p_08 + X_p_08 = 12
    
    _C24: I_p_08 - I_p_09 + X_p_09 = 6
    
    _C25: I_p_09 - I_p_10 + X_p_10 = 8
    
    _C26: I_p_10 - I_p_11 + X_p_11 = 14
    
    _C27: I_p_11 - I_p_12 + X_p_12 = 6
    
    _C28: F_01 - H_01 - W_00 + W_01 = 0
    
    _C29: F_02 - H_02 - W_01 + W_02 = 0
    
    _C30: F_03 - H_03 - W_02 + W_03 = 0
    
    _C31: F_04 - H_04 - W_03 + W_04 = 0
    
    _C32: F_05 - H_05 - W_04 + W_05 = 0
    
    _C33: F_06 - H_06 - W_05 + W_06 = 0
    
    _C34: F_07 - H_07 - W_06 + W_07 = 0
    
    _C35: F_08 - H_08 - W_07 + W_08 = 0
    
    _C36: F_09 - H_09 - W_08 + W_09 = 0
    
    _C37: F_10 - H_10 - W_09 + W_10 = 0
    
    _C38: F_11 - H_11 - W_10 + W_11 = 0
    
    _C39: F_12 - H_12 - W_11 + W_12 = 0
    
    _C40: O_01 - W_01 <= 0
    
    _C41: O_02 - W_02 <= 0
    
    _C42: O_03 - W_03 <= 0
    
    _C43: O_04 - W_04 <= 0
    
    _C44: O_05 - W_05 <= 0
    
    _C45: O_06 - W_06 <= 0
    
    _C46: O_07 - W_07 <= 0
    
    _C47: O_08 - W_08 <= 0
    
    _C48: O_09 - W_09 <= 0
    
    _C49: O_10 - W_10 <= 0
    
    _C50: O_11 - W_11 <= 0
    
    _C51: O_12 - W_12 <= 0
    
    _C52: - 40 O_01 - 180 W_01 + 380 X_b_01 + 530 X_p_01 <= 0
    
    _C53: - 40 O_02 - 180 W_02 + 380 X_b_02 + 530 X_p_02 <= 0
    
    _C54: - 40 O_03 - 180 W_03 + 380 X_b_03 + 530 X_p_03 <= 0
    
    _C55: - 40 O_04 - 180 W_04 + 380 X_b_04 + 530 X_p_04 <= 0
    
    _C56: - 40 O_05 - 180 W_05 + 380 X_b_05 + 530 X_p_05 <= 0
    
    _C57: - 40 O_06 - 180 W_06 + 380 X_b_06 + 530 X_p_06 <= 0
    
    _C58: - 40 O_07 - 180 W_07 + 380 X_b_07 + 530 X_p_07 <= 0
    
    _C59: - 40 O_08 - 180 W_08 + 380 X_b_08 + 530 X_p_08 <= 0
    
    _C60: - 40 O_09 - 180 W_09 + 380 X_b_09 + 530 X_p_09 <= 0
    
    _C61: - 40 O_10 - 180 W_10 + 380 X_b_10 + 530 X_p_10 <= 0
    
    _C62: - 40 O_11 - 180 W_11 + 380 X_b_11 + 530 X_p_11 <= 0
    
    _C63: - 40 O_12 - 180 W_12 + 380 X_b_12 + 530 X_p_12 <= 0
    
    VARIABLES
    F_01 Continuous
    F_02 Continuous
    F_03 Continuous
    F_04 Continuous
    F_05 Continuous
    F_06 Continuous
    F_07 Continuous
    F_08 Continuous
    F_09 Continuous
    F_10 Continuous
    F_11 Continuous
    F_12 Continuous
    H_01 Continuous
    H_02 Continuous
    H_03 Continuous
    H_04 Continuous
    H_05 Continuous
    H_06 Continuous
    H_07 Continuous
    H_08 Continuous
    H_09 Continuous
    H_10 Continuous
    H_11 Continuous
    H_12 Continuous
    I_b_00 Continuous
    I_b_01 Continuous
    I_b_02 Continuous
    I_b_03 Continuous
    I_b_04 Continuous
    I_b_05 Continuous
    I_b_06 Continuous
    I_b_07 Continuous
    I_b_08 Continuous
    I_b_09 Continuous
    I_b_10 Continuous
    I_b_11 Continuous
    I_b_12 Continuous
    I_p_00 Continuous
    I_p_01 Continuous
    I_p_02 Continuous
    I_p_03 Continuous
    I_p_04 Continuous
    I_p_05 Continuous
    I_p_06 Continuous
    I_p_07 Continuous
    I_p_08 Continuous
    I_p_09 Continuous
    I_p_10 Continuous
    I_p_11 Continuous
    I_p_12 Continuous
    O_01 Continuous
    O_02 Continuous
    O_03 Continuous
    O_04 Continuous
    O_05 Continuous
    O_06 Continuous
    O_07 Continuous
    O_08 Continuous
    O_09 Continuous
    O_10 Continuous
    O_11 Continuous
    O_12 Continuous
    W_00 Continuous
    W_01 Continuous
    W_02 Continuous
    W_03 Continuous
    W_04 Continuous
    W_05 Continuous
    W_06 Continuous
    W_07 Continuous
    W_08 Continuous
    W_09 Continuous
    W_10 Continuous
    W_11 Continuous
    W_12 Continuous
    X_b_01 Continuous
    X_b_02 Continuous
    X_b_03 Continuous
    X_b_04 Continuous
    X_b_05 Continuous
    X_b_06 Continuous
    X_b_07 Continuous
    X_b_08 Continuous
    X_b_09 Continuous
    X_b_10 Continuous
    X_b_11 Continuous
    X_b_12 Continuous
    X_p_01 Continuous
    X_p_02 Continuous
    X_p_03 Continuous
    X_p_04 Continuous
    X_p_05 Continuous
    X_p_06 Continuous
    X_p_07 Continuous
    X_p_08 Continuous
    X_p_09 Continuous
    X_p_10 Continuous
    X_p_11 Continuous
    X_p_12 Continuous




```python
model.solve()
LpStatus[model.status]
```




    'Optimal'




```python
print("z* = ", value(model.objective))
```

    z* =  3143976.3829999994
    


```python
for v in model.variables():
    print(v.name, " = ", v.varValue)
```

    F_01  =  11.3889
    F_02  =  2.94444
    F_03  =  0.0
    F_04  =  0.0
    F_05  =  0.0
    F_06  =  0.0
    F_07  =  0.0
    F_08  =  0.0
    F_09  =  0.0
    F_10  =  0.0
    F_11  =  0.0
    F_12  =  0.0
    H_01  =  0.0
    H_02  =  0.0
    H_03  =  0.0
    H_04  =  0.0
    H_05  =  0.0
    H_06  =  0.0
    H_07  =  0.0
    H_08  =  0.0
    H_09  =  4.05833
    H_10  =  0.0
    H_11  =  0.0
    H_12  =  0.0
    I_b_00  =  8.0
    I_b_01  =  0.0
    I_b_02  =  0.0
    I_b_03  =  0.0
    I_b_04  =  0.0
    I_b_05  =  0.0
    I_b_06  =  12.7895
    I_b_07  =  10.7895
    I_b_08  =  8.0
    I_b_09  =  17.5013
    I_b_10  =  10.2132
    I_b_11  =  0.527632
    I_b_12  =  0.0
    I_p_00  =  3.0
    I_p_01  =  0.0
    I_p_02  =  0.0
    I_p_03  =  0.0
    I_p_04  =  0.0
    I_p_05  =  0.0
    I_p_06  =  0.0
    I_p_07  =  0.0
    I_p_08  =  0.0
    I_p_09  =  0.0
    I_p_10  =  0.0
    I_p_11  =  0.0
    I_p_12  =  0.0
    O_01  =  0.0
    O_02  =  0.0
    O_03  =  0.0
    O_04  =  0.0
    O_05  =  0.0
    O_06  =  0.0
    O_07  =  0.0
    O_08  =  0.0
    O_09  =  0.0
    O_10  =  0.0
    O_11  =  75.725
    O_12  =  75.725
    W_00  =  86.0
    W_01  =  74.6111
    W_02  =  71.6667
    W_03  =  71.6667
    W_04  =  71.6667
    W_05  =  71.6667
    W_06  =  71.6667
    W_07  =  71.6667
    W_08  =  71.6667
    W_09  =  75.725
    W_10  =  75.725
    W_11  =  75.725
    W_12  =  75.725
    X_b_01  =  20.0
    X_b_02  =  20.0
    X_b_03  =  26.0
    X_b_04  =  24.0
    X_b_05  =  18.0
    X_b_06  =  22.7895
    X_b_07  =  20.0
    X_b_08  =  17.2105
    X_b_09  =  27.5013
    X_b_10  =  24.7118
    X_b_11  =  24.3145
    X_b_12  =  35.4724
    X_p_01  =  11.0
    X_p_02  =  10.0
    X_p_03  =  4.0
    X_p_04  =  4.0
    X_p_05  =  6.0
    X_p_06  =  8.0
    X_p_07  =  10.0
    X_p_08  =  12.0
    X_p_09  =  6.0
    X_p_10  =  8.0
    X_p_11  =  14.0
    X_p_12  =  6.0
    


```python
production  = []
inventory   = []

for i in range(len(M)):
    production. append([v.varValue for v in model.variables() if ("X" in v.name) & (M[i] in v.name)])
    inventory.  append([v.varValue for v in model.variables() if ("I" in v.name) & (M[i] in v.name)])
            
```


```python
# Plotting the monthly production plan

months = ['Jun22', 'Jul22', 'Aug22', 'Sep22', 'Oct22', 'Nov22',
          'Dec22', 'Jan23', 'Feb23', 'Mar23', 'Apr23', 'May23']

X_axis = np.arange(len(T))          

for i in range(len(production)):
        plt.bar(X_axis - 0.2 + i*0.4, production[i], 0.4, label = "{} Series".format(M[i]))

plt.xticks(X_axis, months, rotation = 90)

plt.xlabel(     "Time Horizon")
plt.ylabel(     "Number of Units Planned to Produce")
plt.title(      "Monthly Production Rates")

plt.legend()
plt.show()


```


    
![png](/assets/img/content/220327/output_12_0.png)
    



```python
# Plotting the monthly inventory levels

X_axis = np.arange(len(T))          

for i in range(len(inventory)):
        plt.bar(X_axis - 0.2 + i*0.4, inventory[i][1:], 0.4, label = "{} Series".format(M[i]))

plt.xticks(X_axis, months, rotation = 90)

plt.xlabel(     "Time Horizon")
plt.ylabel(     "Inventory Level")
plt.title(      "Monthly Inventory Levels")

plt.legend()
plt.show()


```


    
![png](/assets/img/content/220327/output_13_0.png)

## Extension 1

During the months of December and January you have the option increasing your
regular man-hour capacity by bringing temporary skilled workers from another plant.
Therefore, there is no hiring cost. Including the relocation cost, the total cost of extra
labour force will be $15/hour.

## Solution (Extension 1)

```python
from    pulp    import *         

```


```python
# Problem

model = LpProblem(  name    = "aggregate-planning", 
                    sense   = LpMinimize            )
                    
```


```python
# Sets

T       = range(1,13)                   
T_ext   = range(13)                    
M       = ("b", "p")                    

```


```python
# New set: index of the months during which extra workforce can be brought

EM = [7, 8]     # December and January

```


```python
# Parameters

D = (   (28, 14), 
        (20, 10),
        (26, 4),
        (24, 4),
        (18, 6),
        (10, 8),
        (22, 10),
        (20, 12),
        (18, 6),
        (32, 8),
        (34, 14),
        (36, 6)     )           

wInit   = 86                    
h       = 800                   
f       = 1400                  

c       = (250, 500)            
iInit   = (8, 3)                

pCost   = (6250, 9750)          

pWf     = (380, 530)           
p       = 180                  

r       = 420                  
o       = (420/180)*1.5*40     

oLim    = 40
                    
```


```python
# New parameter: cost of bringing temporary workers

e = 15 # [$ / hours]

```


```python
# Decision variables

X = {   (i, t): LpVariable( name="X-{model}-{month}".format(    model   = M[i],                                     
                                                                month   = t     if t>=10 
                                                                                else "0{}".format(t)),              
                            lowBound=0                                                                 )
                                                                                for i in range(len(M)) for t in T       }


I = {   (i, t): LpVariable( name="I-{model}-{month}".format(    model   = M[i],                                     
                                                                month   =   t   if t>=10 
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for i in range(len(M)) for t in T_ext   }


W = {   (t): LpVariable(    name="W-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T_ext                          }


H = {   (t): LpVariable(    name="H-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


F = {   (t): LpVariable(    name="F-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


O = {   (t): LpVariable(    name="O-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }

```


```python
# New decision variable: # of extra working hours supplied by temporary workers

E = {   (t): LpVariable(    name="E-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in EM                          }
                                                                                
```


```python
# Modified objective function

model += lpSum(
                [   pCost[i]*X[(i,t)]   +                                       
                    c[i]*I[(i,t)]                                               
                                            for t in T                          
                                            for i in range(len(M)   )   ] +      
                [   r*W[t] + o*O[t]     +                                       
                    h*H[t] + f*F[t]                                                  
                                            for t in T                  ] +
                [   e*E[t]                                                
                                            for t in EM                 ]           # Extra workforce costs                           
                                                                            )

```


```python
# Constraints


model += (W[0]      ==  wInit                                                                                                    
                                                                                )


for i in range(len(M)):                                                                         
    model += (I[(i, 0)] ==  iInit[i]
                                                                                )


for i in range(len(M)):                                                                         
    for t in T:
        model += (  I[(i,t-1)] + X[(i,t)] == I[(i,t)] + D[t-1][i]
                                                                                )


for t in T:                                                                                    
    model += (  W[t] == W[t-1] + H[t] - F[t]
                                                                                )


for t in T:                                                                                     
    model += (  O[t] <= W[t]
                                                                                )            

```


```python
# Modified capacity constraint

for t in T:                                                                                     
    if t in EM:
        model += (  lpSum(X[(i,t)]*pWf[i] for i in range(len(M))) <= p*W[t] + oLim*O[t] + E[t]                             
                                                                                                )
    else:
        model += (  lpSum(X[(i,t)]*pWf[i] for i in range(len(M))) <= p*W[t] + oLim*O[t]     
                                                                                                )
                                                                                                
```


```python
model
```




    aggregate-planning:
    MINIMIZE
    15*E_07 + 15*E_08 + 1400*F_01 + 1400*F_02 + 1400*F_03 + 1400*F_04 + 1400*F_05 + 1400*F_06 + 1400*F_07 + 1400*F_08 + 1400*F_09 + 1400*F_10 + 1400*F_11 + 1400*F_12 + 800*H_01 + 800*H_02 + 800*H_03 + 800*H_04 + 800*H_05 + 800*H_06 + 800*H_07 + 800*H_08 + 800*H_09 + 800*H_10 + 800*H_11 + 800*H_12 + 250*I_b_01 + 250*I_b_02 + 250*I_b_03 + 250*I_b_04 + 250*I_b_05 + 250*I_b_06 + 250*I_b_07 + 250*I_b_08 + 250*I_b_09 + 250*I_b_10 + 250*I_b_11 + 250*I_b_12 + 500*I_p_01 + 500*I_p_02 + 500*I_p_03 + 500*I_p_04 + 500*I_p_05 + 500*I_p_06 + 500*I_p_07 + 500*I_p_08 + 500*I_p_09 + 500*I_p_10 + 500*I_p_11 + 500*I_p_12 + 140.0*O_01 + 140.0*O_02 + 140.0*O_03 + 140.0*O_04 + 140.0*O_05 + 140.0*O_06 + 140.0*O_07 + 140.0*O_08 + 140.0*O_09 + 140.0*O_10 + 140.0*O_11 + 140.0*O_12 + 420*W_01 + 420*W_02 + 420*W_03 + 420*W_04 + 420*W_05 + 420*W_06 + 420*W_07 + 420*W_08 + 420*W_09 + 420*W_10 + 420*W_11 + 420*W_12 + 6250*X_b_01 + 6250*X_b_02 + 6250*X_b_03 + 6250*X_b_04 + 6250*X_b_05 + 6250*X_b_06 + 6250*X_b_07 + 6250*X_b_08 + 6250*X_b_09 + 6250*X_b_10 + 6250*X_b_11 + 6250*X_b_12 + 9750*X_p_01 + 9750*X_p_02 + 9750*X_p_03 + 9750*X_p_04 + 9750*X_p_05 + 9750*X_p_06 + 9750*X_p_07 + 9750*X_p_08 + 9750*X_p_09 + 9750*X_p_10 + 9750*X_p_11 + 9750*X_p_12 + 0.0
    SUBJECT TO
    _C1: W_00 = 86
    
    _C2: I_b_00 = 8
    
    _C3: I_p_00 = 3
    
    _C4: I_b_00 - I_b_01 + X_b_01 = 28
    
    _C5: I_b_01 - I_b_02 + X_b_02 = 20
    
    _C6: I_b_02 - I_b_03 + X_b_03 = 26
    
    _C7: I_b_03 - I_b_04 + X_b_04 = 24
    
    _C8: I_b_04 - I_b_05 + X_b_05 = 18
    
    _C9: I_b_05 - I_b_06 + X_b_06 = 10
    
    _C10: I_b_06 - I_b_07 + X_b_07 = 22
    
    _C11: I_b_07 - I_b_08 + X_b_08 = 20
    
    _C12: I_b_08 - I_b_09 + X_b_09 = 18
    
    _C13: I_b_09 - I_b_10 + X_b_10 = 32
    
    _C14: I_b_10 - I_b_11 + X_b_11 = 34
    
    _C15: I_b_11 - I_b_12 + X_b_12 = 36
    
    _C16: I_p_00 - I_p_01 + X_p_01 = 14
    
    _C17: I_p_01 - I_p_02 + X_p_02 = 10
    
    _C18: I_p_02 - I_p_03 + X_p_03 = 4
    
    _C19: I_p_03 - I_p_04 + X_p_04 = 4
    
    _C20: I_p_04 - I_p_05 + X_p_05 = 6
    
    _C21: I_p_05 - I_p_06 + X_p_06 = 8
    
    _C22: I_p_06 - I_p_07 + X_p_07 = 10
    
    _C23: I_p_07 - I_p_08 + X_p_08 = 12
    
    _C24: I_p_08 - I_p_09 + X_p_09 = 6
    
    _C25: I_p_09 - I_p_10 + X_p_10 = 8
    
    _C26: I_p_10 - I_p_11 + X_p_11 = 14
    
    _C27: I_p_11 - I_p_12 + X_p_12 = 6
    
    _C28: F_01 - H_01 - W_00 + W_01 = 0
    
    _C29: F_02 - H_02 - W_01 + W_02 = 0
    
    _C30: F_03 - H_03 - W_02 + W_03 = 0
    
    _C31: F_04 - H_04 - W_03 + W_04 = 0
    
    _C32: F_05 - H_05 - W_04 + W_05 = 0
    
    _C33: F_06 - H_06 - W_05 + W_06 = 0
    
    _C34: F_07 - H_07 - W_06 + W_07 = 0
    
    _C35: F_08 - H_08 - W_07 + W_08 = 0
    
    _C36: F_09 - H_09 - W_08 + W_09 = 0
    
    _C37: F_10 - H_10 - W_09 + W_10 = 0
    
    _C38: F_11 - H_11 - W_10 + W_11 = 0
    
    _C39: F_12 - H_12 - W_11 + W_12 = 0
    
    _C40: O_01 - W_01 <= 0
    
    _C41: O_02 - W_02 <= 0
    
    _C42: O_03 - W_03 <= 0
    
    _C43: O_04 - W_04 <= 0
    
    _C44: O_05 - W_05 <= 0
    
    _C45: O_06 - W_06 <= 0
    
    _C46: O_07 - W_07 <= 0
    
    _C47: O_08 - W_08 <= 0
    
    _C48: O_09 - W_09 <= 0
    
    _C49: O_10 - W_10 <= 0
    
    _C50: O_11 - W_11 <= 0
    
    _C51: O_12 - W_12 <= 0
    
    _C52: - 40 O_01 - 180 W_01 + 380 X_b_01 + 530 X_p_01 <= 0
    
    _C53: - 40 O_02 - 180 W_02 + 380 X_b_02 + 530 X_p_02 <= 0
    
    _C54: - 40 O_03 - 180 W_03 + 380 X_b_03 + 530 X_p_03 <= 0
    
    _C55: - 40 O_04 - 180 W_04 + 380 X_b_04 + 530 X_p_04 <= 0
    
    _C56: - 40 O_05 - 180 W_05 + 380 X_b_05 + 530 X_p_05 <= 0
    
    _C57: - 40 O_06 - 180 W_06 + 380 X_b_06 + 530 X_p_06 <= 0
    
    _C58: - E_07 - 40 O_07 - 180 W_07 + 380 X_b_07 + 530 X_p_07 <= 0
    
    _C59: - E_08 - 40 O_08 - 180 W_08 + 380 X_b_08 + 530 X_p_08 <= 0
    
    _C60: - 40 O_09 - 180 W_09 + 380 X_b_09 + 530 X_p_09 <= 0
    
    _C61: - 40 O_10 - 180 W_10 + 380 X_b_10 + 530 X_p_10 <= 0
    
    _C62: - 40 O_11 - 180 W_11 + 380 X_b_11 + 530 X_p_11 <= 0
    
    _C63: - 40 O_12 - 180 W_12 + 380 X_b_12 + 530 X_p_12 <= 0
    
    VARIABLES
    E_07 Continuous
    E_08 Continuous
    F_01 Continuous
    F_02 Continuous
    F_03 Continuous
    F_04 Continuous
    F_05 Continuous
    F_06 Continuous
    F_07 Continuous
    F_08 Continuous
    F_09 Continuous
    F_10 Continuous
    F_11 Continuous
    F_12 Continuous
    H_01 Continuous
    H_02 Continuous
    H_03 Continuous
    H_04 Continuous
    H_05 Continuous
    H_06 Continuous
    H_07 Continuous
    H_08 Continuous
    H_09 Continuous
    H_10 Continuous
    H_11 Continuous
    H_12 Continuous
    I_b_00 Continuous
    I_b_01 Continuous
    I_b_02 Continuous
    I_b_03 Continuous
    I_b_04 Continuous
    I_b_05 Continuous
    I_b_06 Continuous
    I_b_07 Continuous
    I_b_08 Continuous
    I_b_09 Continuous
    I_b_10 Continuous
    I_b_11 Continuous
    I_b_12 Continuous
    I_p_00 Continuous
    I_p_01 Continuous
    I_p_02 Continuous
    I_p_03 Continuous
    I_p_04 Continuous
    I_p_05 Continuous
    I_p_06 Continuous
    I_p_07 Continuous
    I_p_08 Continuous
    I_p_09 Continuous
    I_p_10 Continuous
    I_p_11 Continuous
    I_p_12 Continuous
    O_01 Continuous
    O_02 Continuous
    O_03 Continuous
    O_04 Continuous
    O_05 Continuous
    O_06 Continuous
    O_07 Continuous
    O_08 Continuous
    O_09 Continuous
    O_10 Continuous
    O_11 Continuous
    O_12 Continuous
    W_00 Continuous
    W_01 Continuous
    W_02 Continuous
    W_03 Continuous
    W_04 Continuous
    W_05 Continuous
    W_06 Continuous
    W_07 Continuous
    W_08 Continuous
    W_09 Continuous
    W_10 Continuous
    W_11 Continuous
    W_12 Continuous
    X_b_01 Continuous
    X_b_02 Continuous
    X_b_03 Continuous
    X_b_04 Continuous
    X_b_05 Continuous
    X_b_06 Continuous
    X_b_07 Continuous
    X_b_08 Continuous
    X_b_09 Continuous
    X_b_10 Continuous
    X_b_11 Continuous
    X_b_12 Continuous
    X_p_01 Continuous
    X_p_02 Continuous
    X_p_03 Continuous
    X_p_04 Continuous
    X_p_05 Continuous
    X_p_06 Continuous
    X_p_07 Continuous
    X_p_08 Continuous
    X_p_09 Continuous
    X_p_10 Continuous
    X_p_11 Continuous
    X_p_12 Continuous




```python
model.solve()
LpStatus[model.status]
```




    'Optimal'




```python
print("z* = ", value(model.objective))
```

    z* =  3143976.3829999994
    


```python
for v in model.variables():
    print(v.name, " = ", v.varValue)
```

    E_07  =  0.0
    E_08  =  0.0
    F_01  =  11.3889
    F_02  =  2.94444
    F_03  =  0.0
    F_04  =  0.0
    F_05  =  0.0
    F_06  =  0.0
    F_07  =  0.0
    F_08  =  0.0
    F_09  =  0.0
    F_10  =  0.0
    F_11  =  0.0
    F_12  =  0.0
    H_01  =  0.0
    H_02  =  0.0
    H_03  =  0.0
    H_04  =  0.0
    H_05  =  0.0
    H_06  =  0.0
    H_07  =  0.0
    H_08  =  0.0
    H_09  =  4.05833
    H_10  =  0.0
    H_11  =  0.0
    H_12  =  0.0
    I_b_00  =  8.0
    I_b_01  =  0.0
    I_b_02  =  0.0
    I_b_03  =  0.0
    I_b_04  =  0.0
    I_b_05  =  0.0
    I_b_06  =  12.7895
    I_b_07  =  10.7895
    I_b_08  =  8.0
    I_b_09  =  17.5013
    I_b_10  =  10.2132
    I_b_11  =  0.527632
    I_b_12  =  0.0
    I_p_00  =  3.0
    I_p_01  =  0.0
    I_p_02  =  0.0
    I_p_03  =  0.0
    I_p_04  =  0.0
    I_p_05  =  0.0
    I_p_06  =  0.0
    I_p_07  =  0.0
    I_p_08  =  0.0
    I_p_09  =  0.0
    I_p_10  =  0.0
    I_p_11  =  0.0
    I_p_12  =  0.0
    O_01  =  0.0
    O_02  =  0.0
    O_03  =  0.0
    O_04  =  0.0
    O_05  =  0.0
    O_06  =  0.0
    O_07  =  0.0
    O_08  =  0.0
    O_09  =  0.0
    O_10  =  0.0
    O_11  =  75.725
    O_12  =  75.725
    W_00  =  86.0
    W_01  =  74.6111
    W_02  =  71.6667
    W_03  =  71.6667
    W_04  =  71.6667
    W_05  =  71.6667
    W_06  =  71.6667
    W_07  =  71.6667
    W_08  =  71.6667
    W_09  =  75.725
    W_10  =  75.725
    W_11  =  75.725
    W_12  =  75.725
    X_b_01  =  20.0
    X_b_02  =  20.0
    X_b_03  =  26.0
    X_b_04  =  24.0
    X_b_05  =  18.0
    X_b_06  =  22.7895
    X_b_07  =  20.0
    X_b_08  =  17.2105
    X_b_09  =  27.5013
    X_b_10  =  24.7118
    X_b_11  =  24.3145
    X_b_12  =  35.4724
    X_p_01  =  11.0
    X_p_02  =  10.0
    X_p_03  =  4.0
    X_p_04  =  4.0
    X_p_05  =  6.0
    X_p_06  =  8.0
    X_p_07  =  10.0
    X_p_08  =  12.0
    X_p_09  =  6.0
    X_p_10  =  8.0
    X_p_11  =  14.0
    X_p_12  =  6.0

## Extension 2

The gross space requirements of a basic model needs 40 sq meters for storage,
whereas a pro model needs 60 sq meters in the finished goods park of the company.
The company has total parking area of 500 sq meters. Considering this space constraint
would you revise your aggregate plan? The company also has the option of renting
extra parking space for a fee of $1 per sq meter per month. Would you consider making
a rental agreement, and if so for which months?

Production Engineering and Work Study Department warns you that the standard
hours are measured with an error of 10% (i.e., all labor requirement values can change
by 10%). Assuming that the storage constraint is active, how sensitive is your
optimal plan for scenario to changes in the estimation of labor
requirements? Interpret your findings.

## Solution (Extension 2)

```python
# Import libraries

from    pulp    import *  
import  pandas  as pd        # Will be used to generate and export tables                        

```


```python
# Problem

model = LpProblem(  name    = "aggregate-planning", 
                    sense   = LpMinimize            )
                    
```


```python
# Sets

T       = range(1,13)                   
T_ext   = range(13)                     
M       = ("b", "p")                    

```


```python
# Parameters

D = (   (28, 14), 
        (20, 10),
        (26, 4),
        (24, 4),
        (18, 6),
        (10, 8),
        (22, 10),
        (20, 12),
        (18, 6),
        (32, 8),
        (34, 14),
        (36, 6)     )           

wInit   = 86                    
h       = 800                   
f       = 1400                  

c       = (250, 500)            
iInit   = (8, 3)                

pCost   = (6250, 9750)          

pWf     = (380, 530)            
p       = 180                   

r       = 420                   
o       = (420/180)*1.5*40      

oLim    = 40                    

```


```python
# New parameters

e       = 1             # Cost of renting extra parking space       [$ per sq meter]
gp      = 500           # Total capacity of company's goods park    [sq meters]   
sr      = (40, 60)      # Space requirements                        [sq meters]
```


```python
# Decision variables

X = {   (i, t): LpVariable( name="X-{model}-{month}".format(    model   = M[i],                                     
                                                                month   = t     if t>=10 
                                                                                else "0{}".format(t)),              
                            lowBound=0                                                                 )
                                                                                for i in range(len(M)) for t in T       }


I = {   (i, t): LpVariable( name="I-{model}-{month}".format(    model   = M[i],                                     
                                                                month   =   t   if t>=10 
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for i in range(len(M)) for t in T_ext   }


W = {   (t): LpVariable(    name="W-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T_ext                          }


H = {   (t): LpVariable(    name="H-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


F = {   (t): LpVariable(    name="F-{}"             .format(                t   if t>=10                           
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


O = {   (t): LpVariable(    name="O-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }


```


```python
# New decision variable: amount of extra space rented

E = {   (t): LpVariable(    name="E-{}"             .format(                t   if t>=10                            
                                                                                else "0{}".format(t)),
                            lowBound=0                                                               )              
                                                                                for t in T                              }
```


```python
# Modified objective function

model += lpSum(
                [   pCost[i]*X[(i,t)]   +                                    
                    c[i]*I[(i,t)]                                            
                                            for t in T                       
                                            for i in range(len(M)   )   ] +  
                [   r*W[t] + o*O[t]     +                                    
                    h*H[t] + f*F[t]     +
                    e*E[t]                                          
                                            for t in T                  ]    
                                                                            )
```


```python
# Constraints


model += (W[0]      ==  wInit                                                                               
                                                                                    )


for i in range(len(M)):                                                                         
    model += (I[(i, 0)] ==  iInit[i]
                                                                                    )


for i in range(len(M)):                                                                         
    for t in T:
        model += (  I[(i,t-1)] + X[(i,t)] == I[(i,t)] + D[t-1][i]
                                                                                    )


for t in T:                                                                                     
    model += (  W[t] == W[t-1] + H[t] - F[t]
                                                                                    )


for t in T:                                                                                     
    model += (  O[t] <= W[t]
                                                                                    )           


for t in T:                                                                                     
    model += (  lpSum(X[(i,t)]*pWf[i] for i in range(len(M))) <= p*W[t] + oLim*O[t]             
                                                                                    )

```


```python
# New constraint: Goods park capacity

for t in T:                                                                                     
    model += (  lpSum(X[(i,t)]*sr[i]  for i in range(len(M))) <= gp     + E[t]
                                                                                    )
```


```python
model
```




    aggregate-planning:
    MINIMIZE
    1*E_01 + 1*E_02 + 1*E_03 + 1*E_04 + 1*E_05 + 1*E_06 + 1*E_07 + 1*E_08 + 1*E_09 + 1*E_10 + 1*E_11 + 1*E_12 + 1400*F_01 + 1400*F_02 + 1400*F_03 + 1400*F_04 + 1400*F_05 + 1400*F_06 + 1400*F_07 + 1400*F_08 + 1400*F_09 + 1400*F_10 + 1400*F_11 + 1400*F_12 + 800*H_01 + 800*H_02 + 800*H_03 + 800*H_04 + 800*H_05 + 800*H_06 + 800*H_07 + 800*H_08 + 800*H_09 + 800*H_10 + 800*H_11 + 800*H_12 + 250*I_b_01 + 250*I_b_02 + 250*I_b_03 + 250*I_b_04 + 250*I_b_05 + 250*I_b_06 + 250*I_b_07 + 250*I_b_08 + 250*I_b_09 + 250*I_b_10 + 250*I_b_11 + 250*I_b_12 + 500*I_p_01 + 500*I_p_02 + 500*I_p_03 + 500*I_p_04 + 500*I_p_05 + 500*I_p_06 + 500*I_p_07 + 500*I_p_08 + 500*I_p_09 + 500*I_p_10 + 500*I_p_11 + 500*I_p_12 + 140.0*O_01 + 140.0*O_02 + 140.0*O_03 + 140.0*O_04 + 140.0*O_05 + 140.0*O_06 + 140.0*O_07 + 140.0*O_08 + 140.0*O_09 + 140.0*O_10 + 140.0*O_11 + 140.0*O_12 + 420*W_01 + 420*W_02 + 420*W_03 + 420*W_04 + 420*W_05 + 420*W_06 + 420*W_07 + 420*W_08 + 420*W_09 + 420*W_10 + 420*W_11 + 420*W_12 + 6250*X_b_01 + 6250*X_b_02 + 6250*X_b_03 + 6250*X_b_04 + 6250*X_b_05 + 6250*X_b_06 + 6250*X_b_07 + 6250*X_b_08 + 6250*X_b_09 + 6250*X_b_10 + 6250*X_b_11 + 6250*X_b_12 + 9750*X_p_01 + 9750*X_p_02 + 9750*X_p_03 + 9750*X_p_04 + 9750*X_p_05 + 9750*X_p_06 + 9750*X_p_07 + 9750*X_p_08 + 9750*X_p_09 + 9750*X_p_10 + 9750*X_p_11 + 9750*X_p_12 + 0.0
    SUBJECT TO
    _C1: W_00 = 86
    
    _C2: I_b_00 = 8
    
    _C3: I_p_00 = 3
    
    _C4: I_b_00 - I_b_01 + X_b_01 = 28
    
    _C5: I_b_01 - I_b_02 + X_b_02 = 20
    
    _C6: I_b_02 - I_b_03 + X_b_03 = 26
    
    _C7: I_b_03 - I_b_04 + X_b_04 = 24
    
    _C8: I_b_04 - I_b_05 + X_b_05 = 18
    
    _C9: I_b_05 - I_b_06 + X_b_06 = 10
    
    _C10: I_b_06 - I_b_07 + X_b_07 = 22
    
    _C11: I_b_07 - I_b_08 + X_b_08 = 20
    
    _C12: I_b_08 - I_b_09 + X_b_09 = 18
    
    _C13: I_b_09 - I_b_10 + X_b_10 = 32
    
    _C14: I_b_10 - I_b_11 + X_b_11 = 34
    
    _C15: I_b_11 - I_b_12 + X_b_12 = 36
    
    _C16: I_p_00 - I_p_01 + X_p_01 = 14
    
    _C17: I_p_01 - I_p_02 + X_p_02 = 10
    
    _C18: I_p_02 - I_p_03 + X_p_03 = 4
    
    _C19: I_p_03 - I_p_04 + X_p_04 = 4
    
    _C20: I_p_04 - I_p_05 + X_p_05 = 6
    
    _C21: I_p_05 - I_p_06 + X_p_06 = 8
    
    _C22: I_p_06 - I_p_07 + X_p_07 = 10
    
    _C23: I_p_07 - I_p_08 + X_p_08 = 12
    
    _C24: I_p_08 - I_p_09 + X_p_09 = 6
    
    _C25: I_p_09 - I_p_10 + X_p_10 = 8
    
    _C26: I_p_10 - I_p_11 + X_p_11 = 14
    
    _C27: I_p_11 - I_p_12 + X_p_12 = 6
    
    _C28: F_01 - H_01 - W_00 + W_01 = 0
    
    _C29: F_02 - H_02 - W_01 + W_02 = 0
    
    _C30: F_03 - H_03 - W_02 + W_03 = 0
    
    _C31: F_04 - H_04 - W_03 + W_04 = 0
    
    _C32: F_05 - H_05 - W_04 + W_05 = 0
    
    _C33: F_06 - H_06 - W_05 + W_06 = 0
    
    _C34: F_07 - H_07 - W_06 + W_07 = 0
    
    _C35: F_08 - H_08 - W_07 + W_08 = 0
    
    _C36: F_09 - H_09 - W_08 + W_09 = 0
    
    _C37: F_10 - H_10 - W_09 + W_10 = 0
    
    _C38: F_11 - H_11 - W_10 + W_11 = 0
    
    _C39: F_12 - H_12 - W_11 + W_12 = 0
    
    _C40: O_01 - W_01 <= 0
    
    _C41: O_02 - W_02 <= 0
    
    _C42: O_03 - W_03 <= 0
    
    _C43: O_04 - W_04 <= 0
    
    _C44: O_05 - W_05 <= 0
    
    _C45: O_06 - W_06 <= 0
    
    _C46: O_07 - W_07 <= 0
    
    _C47: O_08 - W_08 <= 0
    
    _C48: O_09 - W_09 <= 0
    
    _C49: O_10 - W_10 <= 0
    
    _C50: O_11 - W_11 <= 0
    
    _C51: O_12 - W_12 <= 0
    
    _C52: - 40 O_01 - 180 W_01 + 380 X_b_01 + 530 X_p_01 <= 0
    
    _C53: - 40 O_02 - 180 W_02 + 380 X_b_02 + 530 X_p_02 <= 0
    
    _C54: - 40 O_03 - 180 W_03 + 380 X_b_03 + 530 X_p_03 <= 0
    
    _C55: - 40 O_04 - 180 W_04 + 380 X_b_04 + 530 X_p_04 <= 0
    
    _C56: - 40 O_05 - 180 W_05 + 380 X_b_05 + 530 X_p_05 <= 0
    
    _C57: - 40 O_06 - 180 W_06 + 380 X_b_06 + 530 X_p_06 <= 0
    
    _C58: - 40 O_07 - 180 W_07 + 380 X_b_07 + 530 X_p_07 <= 0
    
    _C59: - 40 O_08 - 180 W_08 + 380 X_b_08 + 530 X_p_08 <= 0
    
    _C60: - 40 O_09 - 180 W_09 + 380 X_b_09 + 530 X_p_09 <= 0
    
    _C61: - 40 O_10 - 180 W_10 + 380 X_b_10 + 530 X_p_10 <= 0
    
    _C62: - 40 O_11 - 180 W_11 + 380 X_b_11 + 530 X_p_11 <= 0
    
    _C63: - 40 O_12 - 180 W_12 + 380 X_b_12 + 530 X_p_12 <= 0
    
    _C64: - E_01 + 40 X_b_01 + 60 X_p_01 <= 500
    
    _C65: - E_02 + 40 X_b_02 + 60 X_p_02 <= 500
    
    _C66: - E_03 + 40 X_b_03 + 60 X_p_03 <= 500
    
    _C67: - E_04 + 40 X_b_04 + 60 X_p_04 <= 500
    
    _C68: - E_05 + 40 X_b_05 + 60 X_p_05 <= 500
    
    _C69: - E_06 + 40 X_b_06 + 60 X_p_06 <= 500
    
    _C70: - E_07 + 40 X_b_07 + 60 X_p_07 <= 500
    
    _C71: - E_08 + 40 X_b_08 + 60 X_p_08 <= 500
    
    _C72: - E_09 + 40 X_b_09 + 60 X_p_09 <= 500
    
    _C73: - E_10 + 40 X_b_10 + 60 X_p_10 <= 500
    
    _C74: - E_11 + 40 X_b_11 + 60 X_p_11 <= 500
    
    _C75: - E_12 + 40 X_b_12 + 60 X_p_12 <= 500
    
    VARIABLES
    E_01 Continuous
    E_02 Continuous
    E_03 Continuous
    E_04 Continuous
    E_05 Continuous
    E_06 Continuous
    E_07 Continuous
    E_08 Continuous
    E_09 Continuous
    E_10 Continuous
    E_11 Continuous
    E_12 Continuous
    F_01 Continuous
    F_02 Continuous
    F_03 Continuous
    F_04 Continuous
    F_05 Continuous
    F_06 Continuous
    F_07 Continuous
    F_08 Continuous
    F_09 Continuous
    F_10 Continuous
    F_11 Continuous
    F_12 Continuous
    H_01 Continuous
    H_02 Continuous
    H_03 Continuous
    H_04 Continuous
    H_05 Continuous
    H_06 Continuous
    H_07 Continuous
    H_08 Continuous
    H_09 Continuous
    H_10 Continuous
    H_11 Continuous
    H_12 Continuous
    I_b_00 Continuous
    I_b_01 Continuous
    I_b_02 Continuous
    I_b_03 Continuous
    I_b_04 Continuous
    I_b_05 Continuous
    I_b_06 Continuous
    I_b_07 Continuous
    I_b_08 Continuous
    I_b_09 Continuous
    I_b_10 Continuous
    I_b_11 Continuous
    I_b_12 Continuous
    I_p_00 Continuous
    I_p_01 Continuous
    I_p_02 Continuous
    I_p_03 Continuous
    I_p_04 Continuous
    I_p_05 Continuous
    I_p_06 Continuous
    I_p_07 Continuous
    I_p_08 Continuous
    I_p_09 Continuous
    I_p_10 Continuous
    I_p_11 Continuous
    I_p_12 Continuous
    O_01 Continuous
    O_02 Continuous
    O_03 Continuous
    O_04 Continuous
    O_05 Continuous
    O_06 Continuous
    O_07 Continuous
    O_08 Continuous
    O_09 Continuous
    O_10 Continuous
    O_11 Continuous
    O_12 Continuous
    W_00 Continuous
    W_01 Continuous
    W_02 Continuous
    W_03 Continuous
    W_04 Continuous
    W_05 Continuous
    W_06 Continuous
    W_07 Continuous
    W_08 Continuous
    W_09 Continuous
    W_10 Continuous
    W_11 Continuous
    W_12 Continuous
    X_b_01 Continuous
    X_b_02 Continuous
    X_b_03 Continuous
    X_b_04 Continuous
    X_b_05 Continuous
    X_b_06 Continuous
    X_b_07 Continuous
    X_b_08 Continuous
    X_b_09 Continuous
    X_b_10 Continuous
    X_b_11 Continuous
    X_b_12 Continuous
    X_p_01 Continuous
    X_p_02 Continuous
    X_p_03 Continuous
    X_p_04 Continuous
    X_p_05 Continuous
    X_p_06 Continuous
    X_p_07 Continuous
    X_p_08 Continuous
    X_p_09 Continuous
    X_p_10 Continuous
    X_p_11 Continuous
    X_p_12 Continuous




```python
model.solve()
LpStatus[model.status]
```




    'Optimal'




```python
print("z* = ", value(model.objective))
```

    z* =  3155116.3799999994
    


```python
for v in model.variables():
    print(v.name, " = ", v.varValue)
```

    E_01  =  960.0
    E_02  =  900.0
    E_03  =  780.0
    E_04  =  700.0
    E_05  =  580.0
    E_06  =  891.579
    E_07  =  900.0
    E_08  =  908.421
    E_09  =  960.053
    E_10  =  968.474
    E_11  =  1312.58
    E_12  =  1278.89
    F_01  =  11.3889
    F_02  =  2.94444
    F_03  =  0.0
    F_04  =  0.0
    F_05  =  0.0
    F_06  =  0.0
    F_07  =  0.0
    F_08  =  0.0
    F_09  =  0.0
    F_10  =  0.0
    F_11  =  0.0
    F_12  =  0.0
    H_01  =  0.0
    H_02  =  0.0
    H_03  =  0.0
    H_04  =  0.0
    H_05  =  0.0
    H_06  =  0.0
    H_07  =  0.0
    H_08  =  0.0
    H_09  =  4.05833
    H_10  =  0.0
    H_11  =  0.0
    H_12  =  0.0
    I_b_00  =  8.0
    I_b_01  =  0.0
    I_b_02  =  0.0
    I_b_03  =  0.0
    I_b_04  =  0.0
    I_b_05  =  0.0
    I_b_06  =  12.7895
    I_b_07  =  10.7895
    I_b_08  =  8.0
    I_b_09  =  17.5013
    I_b_10  =  10.2132
    I_b_11  =  0.527632
    I_b_12  =  0.0
    I_p_00  =  3.0
    I_p_01  =  0.0
    I_p_02  =  0.0
    I_p_03  =  0.0
    I_p_04  =  0.0
    I_p_05  =  0.0
    I_p_06  =  0.0
    I_p_07  =  0.0
    I_p_08  =  0.0
    I_p_09  =  0.0
    I_p_10  =  0.0
    I_p_11  =  0.0
    I_p_12  =  0.0
    O_01  =  0.0
    O_02  =  0.0
    O_03  =  0.0
    O_04  =  0.0
    O_05  =  0.0
    O_06  =  0.0
    O_07  =  0.0
    O_08  =  0.0
    O_09  =  0.0
    O_10  =  0.0
    O_11  =  75.725
    O_12  =  75.725
    W_00  =  86.0
    W_01  =  74.6111
    W_02  =  71.6667
    W_03  =  71.6667
    W_04  =  71.6667
    W_05  =  71.6667
    W_06  =  71.6667
    W_07  =  71.6667
    W_08  =  71.6667
    W_09  =  75.725
    W_10  =  75.725
    W_11  =  75.725
    W_12  =  75.725
    X_b_01  =  20.0
    X_b_02  =  20.0
    X_b_03  =  26.0
    X_b_04  =  24.0
    X_b_05  =  18.0
    X_b_06  =  22.7895
    X_b_07  =  20.0
    X_b_08  =  17.2105
    X_b_09  =  27.5013
    X_b_10  =  24.7118
    X_b_11  =  24.3145
    X_b_12  =  35.4724
    X_p_01  =  11.0
    X_p_02  =  10.0
    X_p_03  =  4.0
    X_p_04  =  4.0
    X_p_05  =  6.0
    X_p_06  =  8.0
    X_p_07  =  10.0
    X_p_08  =  12.0
    X_p_09  =  6.0
    X_p_10  =  8.0
    X_p_11  =  14.0
    X_p_12  =  6.0
    


```python
o = [{'name':name, 'shadow price':c.pi, 'slack': c.slack} 
     for name, c in model.constraints.items()]
print(pd.DataFrame(o))

```

        name shadow price slack
    0    _C1         None  None
    1    _C2         None  None
    2    _C3         None  None
    3    _C4         None  None
    4    _C5         None  None
    ..   ...          ...   ...
    70  _C71         None  None
    71  _C72         None  None
    72  _C73         None  None
    73  _C74         None  None
    74  _C75         None  None
    
    [75 rows x 3 columns]


*Solution by Ahmet Yiğit Doğan*  
*IE 313 - Supply Chain Management*  
*Boğaziçi University - Industrial Engineering Department*
[GitHub Repository](https://github.com/ayigitdogan/Supply-Chain-Case-Study-1-Aggregate-Planning)
