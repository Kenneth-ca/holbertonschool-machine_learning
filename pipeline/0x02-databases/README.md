# 0x02. Databases

After fetching data via APIs, storing them is also really important for training a Machine Learning model.

You have multiple option:

* Relation database
* Not Relation database
* Key-Value storage
* Document storage
* Data Lake
* etc.

In this project, you will touch the first 2: relation and not relation database.

Relation databases are mainly used for application, not for source of data for training your ML models, but it can be really useful for the data processing, labeling and injection in another data storage. In this project, you will play with basic SQL commands but also create automation and computing on your data directly in SQL - less load at your application level since the computing power is dispatched to the database.

Not relation databases, known as NoSQL, will give you flexibility on your data: document, versioning, not a fix schema, no validation to improve performance, complex lookup, etc.

## Learning Objectives

### General
* What’s a relational database
* What’s a none relational database
* What is difference between SQL and NoSQL
* How to create tables with constraints
* How to optimize queries by adding indexes
* What is and how to implement stored procedures and functions in MySQL
* What is and how to implement views in MySQL
* What is and how to implement triggers in MySQL
* What is ACID
* What is a document storage
* What are NoSQL types
* What are benefits of a NoSQL database
* How to query information from a NoSQL database
* How to insert/update/delete information from a NoSQL database
* How to use MongoDB
