# Why are our customers churning?

### Predicting Telecommunication Customer Churn With Classification Models/Algorithms 

### *The Report*

> Here's the link to the deliverable report requested from the team lead. It contains the slides, but not the verbal portion of the presentation. 

https://docs.google.com/presentation/d/1nObaqS4VnmiCM28RdxW_XqJfoU9f-sAFba4UjRf3A00/edit?usp=sharing

## *Background*:

> Our team lead would like us to take a look at some of our recent customer data. We've been tasked with identifying areas that represent high customer churn.

> Aside from the more general question, *why are our customers churning?* Some other questions we will look to answer: Is there a price threshold for specific services where the likelihood of churn increases? Is their a negative impact once the price for those services goes past that point? If so, what is that point for what service(s)? Among numerous other possible questions.

> For this particular project she would like to see our code documentation and commenting buttoned-up. In addition, she'd like us to not leave any individual numbers or figures displayed in isolation. Adding context to these situations are necessary.

## *Other notes*:

- Other possible questions: Are there clear groupings where a customer is more likely to churn? What if you consider contract type? Is there a tenure that month-to-month customers are most likely to churn? 1-year contract customers? 2-year customers? Do you have any thoughts on what could be going on? 
- Are there features that indicate a higher propensity to churn? Such as type of internet service, type of phone service, online security and backup, senior citizens, paying more than x% of customers with the same services, etc.?
- If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?

# Specification

## *Goals*

To identify as many different customer subgroups that have a propensity to churn more than others. Our target audience is our team lead, however, she will be presenting these findings to the Senior Leadership Team. We will need to keep this final audience in mind with regards to report readability, etc. We will need to communicate in a more concise and clear manner.

## *Deliverables*

**What should our team lead expect to receive from us?**

1. A report (in the form of a presentation, both verbal and through a slides)

She will need a slide deck, with 1-3 slides, that illustrates how our model works, including the features being used, so that she can deliver this to the Senior Leadership Team (SLT) when they come with questions about how these values were derived. She's asked us to please include how likely our model is to give a high probability of churn when churn doesn't occur, to give a low probability of churn when churn occurs, and to accurately predict churn.

2. A github repository containing our jupyter notebook that walks through the pipeline along with the .py files necessary to reproduce our model. The data dictionary can be found in this file.

***model must be reproducible by someone with their own env.py file***

3. She will also need a csv with the customer_id, probability of churn, and the prediction of churn (1=churn, 0=not_churn) - a csv file that predicts churn for each customer.

4. Finally, the development team will need a .py file that will take in a new dataset, and perform all the transformations necessary to run the model we have developed on this new dataset to provide probabilities and predictions.

## *The Pipeline*

### PROJECT PLANNING & README

> Brainstorming ideas, hypotheses, related to how variables might impact or relate to each other, both within independent variables and between the independent variables and dependent variable, and also related to any ideas for new features you may have while first looking at the existing variables and challenge ahead of you.

> In addition: we will summarize our project and goals. We will task out how we will work through the pipeline, in as much detail as we need to keep on track.

### ACQUIRE:

**Goal**: leave this section with a dataframe ready to prepare.

The ad hoc part includes summarizing your data as you read it in and begin to explore, look at the first few rows, data types, summary stats, column names, shape of the data frame, etc.

acquire.py: The reproducible part is the gathering data from SQL.

### PREP:

**Goal**: leave this section with a dataset that is ready to be analyzed. Data types are appropriate, missing values have been addressed, as have any data integrity issues.

The ad hoc part includes plotting the distributions of individual variables and using those plots to identify outliers and if those should be handled (and if so, how), identify unit scales to identify how to best scale the numeric data, as well as finding erroneous or invalid data that may exist in your dataframe.

Some items to consider:

- [X] split data to train/test
- [X] Handle Missing Values
- [X] Handle erroneous data and/or outliers you wish to address
- [X] encode variables as needed
- [X] scale data as needed
- [X] new feature that represents tenure in years
- [X] create single variable representing the information from phone_service and multiple_lines
- [X] do the same using dependents and partner
- [X] other ways to merge variables, such as streaming_tv & streaming_movies, online_security & online_backup

prep.py: The reproducible part is the handling of missing values, fixing data integrity issues, changing data types, etc.

### DATA EXPLORATION & FEATURE SELECTION

**Goal**: Address each of the questions posed in our planning and brainstorming phase - as time permits. As well as any uncovered that come up during the visual or statistical analysis.

When you have completed this step, we will have the findings from our analysis that will be used in the final report, answers to specific questions the stakeholders asked, and information to move forward toward building a model.

Answer the key questions, our hypotheses, and figure out the drivers of churn.

1. If a group is identified by tenure, is there a cohort or cohorts who have a higher rate of churn than other cohorts? (Plot the rate of churn on a line chart where x is the tenure and y is the rate of churn (customers churned/total customers))

2. Are there features that indicate a higher propensity to churn? like type of internet service, type of phone service, online security and backup, senior citizens, paying more than x% of customers with the same services, etc.?

3. Is there a price threshold for specific services where the likelihood of churn increases once price for those services goes past that point? If so, what is that point for what service(s)?

4. If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?

5. Create visualizations exploring the interactions of variables (independent with independent and independent with dependent). The goal is to identify features that are related to churn, identify any data integrity issues, understand 'how the data works'.

6. What can you say about each variable's relationship to churn, based on your initial exploration? If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.

7. Summarize your conclusions, provide clear answers to the specific questions, and summarize any takeaways/action plan from the work above.

### FEATURE SELECTION

**Goal**: leave this section with a dataframe with the features to be used to build our model.

Are there new features we could create based on existing features that might be helpful?

We could use feature selection techniques to see if there are any that are not adding value to the model.

feature_selection.py: to run whatever functions need to be run to end with a dataframe that contains the features that will be used to model the data.

### MODELING & EVALUATION

**Goal**: develop a classification model that performs better than a baseline.

1. Train (fit, transform, evaluate) multiple different models, varying the model type and your meta-parameters.

2. Compare evaluation metrics across all the models, and select the best performing model.

3. Test the final model (transform, evaluate) on your out-of-sample data (the testing data set). Summarize the performance. Interpret your results.

model.py: will have the functions to fit, predict and evaluate the model

### SUMMARY

## *SQL Data Acquisition*

Must use your own env file to access data.

***

## *Technical Skills used*

* Python
* SQL
* Various data science libraries (Pandas, Numpy, Matplotlib, Sklearn, etc.)
* Stats (Hypothesis testing, correlation tests)
* Classification Models (Logistic Regression, Decision Tree, Random Forest, K Nearest Neighbors)
* PowerPoint (presentation creation)

***

## *Executive Summary*

The model we selected to move forward with is a Classification ML Algorithm, Logistic Regression. 

The model is able to correctly predict 80% of customers who actually churn.

Customers most likely to churn:

- Month to month contract
- Senior Citizens
- Utilize Premium offerings

The initial Prediction CSV is in this repo, the code outlined in the notebook will produce the CSV as well.

Moving forward we should focus our CRM and marketing efforts on transitioning customers away from month-to-month contracts, strengthen our relationships with our Senior citizen customers, and further utilize our premium services to retain customers.