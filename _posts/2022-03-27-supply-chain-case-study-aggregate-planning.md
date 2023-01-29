---
title: Supply Chain Case Study 1 - Aggregate Planning
date: 2022-03-27 14:10:00 +0800
categories: [Optimization]
tags: [linear programming, pulp, python, supply chain]
render_with_liquid: false
---

*(Important note: To get the most out of the below case study, [the solution report](/assets/pdf/Report%20-%20Supply%20Chain%20Case%20Study%201%20-%20Aggregate%20Planning.pdf),
which includes LP formulations, optimal values of the decision variables,
and interpretations of the results, can be used as a companion.)*

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
model.solve()
LpStatus[model.status]
```




    'Optimal'




```python
print("z* = ", value(model.objective))
```

    z* =  3143976.3829999994
    

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
model.solve()
LpStatus[model.status]
```




    'Optimal'




```python
print("z* = ", value(model.objective))
```

    z* =  3143976.3829999994
    

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
model.solve()
LpStatus[model.status]
```




    'Optimal'




```python
print("z* = ", value(model.objective))
```

    z* =  3155116.3799999994
    


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
*[GitHub Repository](https://github.com/ayigitdogan/Supply-Chain-Case-Study-1-Aggregate-Planning)*
