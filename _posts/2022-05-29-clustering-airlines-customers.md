---
title: Clustering Airlines Customers
date: 2022-05-29 14:10:00 +0800
categories: [Engineering, Machine Learning]
tags: [clustering, hierarchical clustering, k-means clustering, r]
render_with_liquid: false
---

## Introduction

East-West Airlines is trying to learn more about its customers. Key
issues are their flying patterns, earning and use of frequent flyer
rewards, and use of the airline credit card. The task is to identify
customer segments via clustering. The file *EastWestAirlines.xlsx*
contains information on 4000 passengers who belong to an airline’s
frequent flier program. For each passenger the data include information
on their mileage history and on different ways they accrued or spent
miles in the last year. The goal is to try to identify clusters of
passengers that have similar characteristics for the purpose of
targeting different segments for different types of mileage offers.

``` r
# Library imports

library(readxl)
library(dplyr)
library(ggplot2)
library(cluster)

# Setting the seed

seed <- 425
```

``` r
# Importing, modifying, and checking the data set

data        <- read_excel("EastWestAirlines.xlsx", sheet = "data")

data        <- data[,-1]                # Dropping the unnecessary index column

data.scaled <- apply(data,              # Scaling the data between 0 and 1
                     MARGIN = 2,
                     FUN = function(X) (X - min(X))/diff(range(X)))

summary(data.scaled)                    # Checking the modified version
```

    ##     Balance          Qual_miles        cc1_miles        cc2_miles       
    ##  Min.   :0.00000   Min.   :0.00000   Min.   :0.0000   Min.   :0.000000  
    ##  1st Qu.:0.01087   1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.000000  
    ##  Median :0.02528   Median :0.00000   Median :0.0000   Median :0.000000  
    ##  Mean   :0.04317   Mean   :0.01293   Mean   :0.2649   Mean   :0.007252  
    ##  3rd Qu.:0.05420   3rd Qu.:0.00000   3rd Qu.:0.5000   3rd Qu.:0.000000  
    ##  Max.   :1.00000   Max.   :1.00000   Max.   :1.0000   Max.   :1.000000  
    ##    cc3_miles         Bonus_miles       Bonus_trans      Flight_miles_12mo
    ##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
    ##  1st Qu.:0.000000   1st Qu.:0.00474   1st Qu.:0.03488   1st Qu.:0.00000  
    ##  Median :0.000000   Median :0.02720   Median :0.13953   Median :0.00000  
    ##  Mean   :0.003063   Mean   :0.06502   Mean   :0.13491   Mean   :0.01493  
    ##  3rd Qu.:0.000000   3rd Qu.:0.09026   3rd Qu.:0.19767   3rd Qu.:0.01009  
    ##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
    ##  Flight_trans_12   Days_since_enroll     Award       
    ##  Min.   :0.00000   Min.   :0.0000    Min.   :0.0000  
    ##  1st Qu.:0.00000   1st Qu.:0.2807    1st Qu.:0.0000  
    ##  Median :0.00000   Median :0.4936    Median :0.0000  
    ##  Mean   :0.02592   Mean   :0.4963    Mean   :0.3703  
    ##  3rd Qu.:0.01887   3rd Qu.:0.6979    3rd Qu.:1.0000  
    ##  Max.   :1.00000   Max.   :1.0000    Max.   :1.0000

## Hierarchical Clustering

Applying hierarchical clustering with Euclidean distance and complete
linkage:

``` r
# Applying hierarchical clustering

dm_a            <- dist(data.scaled,                    
                        method      = "euclidean")  # Dissimilarity matrix (Euclidean distance)

hc.complete.a   <- hclust(dm_a,
                          method    = "complete")   # Hierarchical clustering using complete linkage

sil.widths.a    <- c()                              # Empty vector to store silhouette widths
                                                    # for different values of K

for (i in 2:10) {                                   # The for loop that generates cluster sets and
                                                    # calculates their respective silhouette widths
    
    clust               <- cutree(hc.complete.a, k = i)
    
    sil                 <- silhouette(clust, dm_a)
    
    sil.widths.a[i-1]   <- mean(sil[, c("sil_width")])    
    
}

results_a <- data.frame(k = 2:10, sil.widths.a)     # Storing the results in a data frame

results_a                                           # Viewing the results
```

    ##    k sil.widths.a
    ## 1  2    0.4704091
    ## 2  3    0.5318787
    ## 3  4    0.4645971
    ## 4  5    0.4167248
    ## 5  6    0.4172319
    ## 6  7    0.4161869
    ## 7  8    0.3912069
    ## 8  9    0.3519030
    ## 9 10    0.3522325

As can be seen in the above table, K = 3 yields the highest silhouette
width, which is 0.53, implying that 3 is the appropriate number of
clusters to be used.

The next step is to compare the cluster centroids to characterize the
different clusters and try to give each cluster a label. To check the
final state of the clusters, the clustering with the best k value can be
performed as follows:

``` r
# Performing the best clustering

hc.best.a     <- cutree(hc.complete.a, 
                        k = results_a[which.max(results_a$sil.widths.a), 1])

table(hc.best.a)
```

    ## hc.best.a
    ##    1    2    3 
    ## 2526 1469    4

The above results indicate that two of the clusters dominates the data
set, while the other one includes only 4 observations.

``` r
# Creating a table that displays the centroids of two clusters

data.w.clusters <- mutate(data, Cluster = hc.best.a)

first.cluster   <- data.w.clusters[data.w.clusters$Cluster == 1, ]

second.cluster  <- data.w.clusters[data.w.clusters$Cluster == 2, ]

third.cluster   <- data.w.clusters[data.w.clusters$Cluster == 3, ]

centroids       <- data.frame("1" = colMeans(first.cluster),
                              "2" = colMeans(second.cluster),
                              "3" = colMeans(third.cluster)     )

print(round(centroids, 2))
```

    ##                         X1       X2        X3
    ## Balance           59791.06 97189.59 131999.50
    ## Qual_miles           88.19   239.73    347.00
    ## cc1_miles             1.70     2.67      2.50
    ## cc2_miles             1.02     1.01      1.00
    ## cc3_miles             1.01     1.01      1.00
    ## Bonus_miles       10324.88 28739.99  65634.25
    ## Bonus_trans           9.19    15.59     69.25
    ## Flight_miles_12mo   230.44   801.79  19960.00
    ## Flight_trans_12       0.67     2.45     49.25
    ## Days_since_enroll  3824.89  4628.76   2200.25
    ## Award                 0.00     1.00      1.00
    ## Cluster               1.00     2.00      3.00

To obtain more insights about the clusters, centroids can be compared to
the quartiles:

``` r
summary(data)
```

    ##     Balance          Qual_miles        cc1_miles      cc2_miles    
    ##  Min.   :      0   Min.   :    0.0   Min.   :1.00   Min.   :1.000  
    ##  1st Qu.:  18528   1st Qu.:    0.0   1st Qu.:1.00   1st Qu.:1.000  
    ##  Median :  43097   Median :    0.0   Median :1.00   Median :1.000  
    ##  Mean   :  73601   Mean   :  144.1   Mean   :2.06   Mean   :1.015  
    ##  3rd Qu.:  92404   3rd Qu.:    0.0   3rd Qu.:3.00   3rd Qu.:1.000  
    ##  Max.   :1704838   Max.   :11148.0   Max.   :5.00   Max.   :3.000  
    ##    cc3_miles      Bonus_miles      Bonus_trans   Flight_miles_12mo
    ##  Min.   :1.000   Min.   :     0   Min.   : 0.0   Min.   :    0.0  
    ##  1st Qu.:1.000   1st Qu.:  1250   1st Qu.: 3.0   1st Qu.:    0.0  
    ##  Median :1.000   Median :  7171   Median :12.0   Median :    0.0  
    ##  Mean   :1.012   Mean   : 17145   Mean   :11.6   Mean   :  460.1  
    ##  3rd Qu.:1.000   3rd Qu.: 23801   3rd Qu.:17.0   3rd Qu.:  311.0  
    ##  Max.   :5.000   Max.   :263685   Max.   :86.0   Max.   :30817.0  
    ##  Flight_trans_12  Days_since_enroll     Award       
    ##  Min.   : 0.000   Min.   :   2      Min.   :0.0000  
    ##  1st Qu.: 0.000   1st Qu.:2330      1st Qu.:0.0000  
    ##  Median : 0.000   Median :4096      Median :0.0000  
    ##  Mean   : 1.374   Mean   :4119      Mean   :0.3703  
    ##  3rd Qu.: 1.000   3rd Qu.:5790      3rd Qu.:1.0000  
    ##  Max.   :53.000   Max.   :8296      Max.   :1.0000

The three obtained clusters seem to be representing customer groups with
different activity rates.

The third cluster have extremely high activity, when the averages of
activity in the last year, bonus usage, and traveled miles of these
customers are considered. Most of the features have higher average than
the third quartile for this group. It can also be said that they are
relatively new customers.

The first cluster consists of regular customers who travel occasionally.
The second cluster have higher bonus rates and traveled distances in the
last year, along with longer relationships with the company.

Based on these insights, the customers in the first, second, third
clusters can be labelled as *usual customers*, *loyal customers*, and
*super customers*, respectively.

To check the stability of the clusters, a random 5% of the data (200
observations) can be removed, and the analysis can be repeated with the
modified data set.

``` r
# Dropping 200 random rows from the data set

set.seed(seed)

drop        <- as.numeric(sample.int(3999, 200, replace = FALSE))

sliced.data <- data.scaled[-drop, ]

# Repeating the analysis

dm_b            <- dist(sliced.data,                    
                        method = "euclidean")       

hc.complete.b   <- hclust(dm_b,
                          method = "complete")      

sil.widths.b    <- c()                              

for (i in 2:10) {                                   
                                                    
    
    clust               <- cutree(hc.complete.b, k = i)
    
    sil                 <- silhouette(clust, dm_b)
    
    sil.widths.b[i-1]   <- mean(sil[, c("sil_width")])    
}

results_b <- data.frame(k = 2:10, sil.widths.b)       

results_b
```

    ##    k sil.widths.b
    ## 1  2    0.4702675
    ## 2  3    0.5338807
    ## 3  4    0.4796374
    ## 4  5    0.3835390
    ## 5  6    0.3782369
    ## 6  7    0.3918081
    ## 7  8    0.3687406
    ## 8  9    0.3698451
    ## 9 10    0.3717370

``` r
# Checking the new clusters

hc.best.b <- cutree(hc.complete.b, 
                    k = results_b[which.max(results_b$sil.widths.b), 1])

table(hc.best.b)
```

    ## hc.best.b
    ##    1    2    3 
    ## 2387 1408    4

Excluding 200 random observations did not cause dramatic changes in the
clustering results. Considering the general non-robustness problem with
clustering methods, it can be concluded that the current model is not
fragile against the changes in the data set and gives useful insights.

## K-Means Clustering

Using K-Means algorithm with different number of clusters and
determining the best number of clusters using the silhouette index:

``` r
# Applying k-means clustering

dm_c            <- dist(data.scaled,
                        method = "euclidean")

sil.widths.c    <- c()

for (i in 2:10) {                                   
    
    clust               <- kmeans(data.scaled, 
                                  centers   = i, 
                                  nstart    = 50)
    
    sil                 <- silhouette(clust$cl, dm_c)
    
    sil.widths.c[i-1]   <- mean(sil[, c("sil_width")])    
}

results_c <- data.frame(k = 2:10, sil.widths.c)

results_c 
```

    ##    k sil.widths.c
    ## 1  2    0.5359305
    ## 2  3    0.4648453
    ## 3  4    0.4748377
    ## 4  5    0.4430672
    ## 5  6    0.4037918
    ## 6  7    0.3909657
    ## 7  8    0.3803502
    ## 8  9    0.3416435
    ## 9 10    0.3452961

``` r
# Checking the clusters obtained by using the best k value

km.best <- kmeans(data.scaled, 
                  centers   = results_c[which.max(results_c$sil.widths.c), 1], 
                  nstart    = 50)

table(km.best$cl)
```

    ## 
    ##    1    2 
    ## 1481 2518

Setting K = 2 yields the highest silhouette width, which is 0.54,
implying that 2 is the appropriate number of clusters to be used. It
seems like K-Means clustering algorithm suggests that super customer is
not an important insight.

## Final Comments

Since major part of the customers fall into the first cluster in
hierarchically clustered data, setting these customers as target
audience is reasonable. Increasing the number of reward point campaigns
might potentially convert these usual customers to loyal customers.
Special campaigns and discounts for the loyal customers would also be
useful to keep these customers loyal, or event convert them to super
customers.

For the clusters obtained by K-Means algorithm, similar strategy can be
used by picking the second cluster as the main target audience due to
the higher number of observations fall into this set. The absence of
super customers is not a decision criteria since their amount is
negligible.
 
*IE 425 - Data Mining*  
*Boğaziçi University - Industrial Engineering Department*  
[GitHub Repository](https://github.com/ayigitdogan/Clustering-Airlines-Customers)
