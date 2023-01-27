---
title: Building an MRP Database Model with MySQL
date: 2023-01-14 14:10:00 +0800
categories: [Database Management]
tags: [database, mrp, mysql, sql]
render_with_liquid: false
---

## Introduction

In this study, a set of SQL calculations are programmed to implement the Material Resources Planning Algorithm (MRP). The queries include a sequence of statements to create a sample database, similar to the trumpet production example from Production and Operations Analysis[^footnote], function definitions, stored procedure definitions, and finally an SQL trigger that takes action after a new order is inserted into the database, all of which can be found in the folder as “.sql” files. The key SQL concepts that were resorted to in the code are as follows:

Functions (to easily reach to item specific information)
Triggers (to make the database take action after an order insertion)
Stored Procedures (to take a sequence of actions after the trigger)
Recursive Common Table Expressions (to update the “required_items” table by inserting a breakdown structure)
Cursors (to update the “item_period” table for each item and each period)

## Workflow of the Program

The database updates itself when a new row is inserted into the “orders” table. Thus, it is the only table to be updated manually, when a new order is received. After insertion, the following actions are taken by database:

The “after_order_insertion” trigger gets alerted. This trigger consists of two parts. The first part calls the procedure “GetRequiredItemsCount” to obtain the breakdown structure mentioned in the previous parts.
By using the breakdown structure, the gross requirements in the table “item_period” are updated.
The second part of the trigger calls the procedure “UpdateMRPTables”, which includes the core MRP calculations.
Inside this procedure, cursors loop through the items and periods to update projected inventories, planned order receipts, and planned order releases “according to the gross requirements” which have been updated in the first part of the trigger.

## Code and Report

For further information (Entity-Relationship diagram, test case results etc.), please refer to the [report](/assets/pdf/Report%20-%20Building%20an%20MRP%20Database%20Model%20with%20MySQL.pdf) of this project. You can check the code that generates the database and the test case code in the [GitHub repository](https://github.com/ayigitdogan/Building-an-MRP-Database-Model-with-MySQL) of the project.

[^footnote]: Nahmias, S., Olsen, T. (2015). Production and Operations Analysis. (pp. 444-445). Waveland Press. Seventh edition.

*Written by Ahmet Yiğit Doğan*  
*IE 442 - Enterprise Information Systems*  
*Boğaziçi University - Industrial Engineering Department*
