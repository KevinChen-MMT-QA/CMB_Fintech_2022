# CMB_Fintech_2022
A LightGBM-based solution for CMB-Fintech Contest 2022 (Track for Data Science). 

The link of competition: https://www.nowcoder.com/activity/2022cmb/index 

In this contest, we need to design a model to predict customer defection rate. Given datasets containing transaction records, our goal is to predict the probability of customers taking out their deposits after three months.

The dataset contains a training set (./data/train.csv) and two test sets (./data/test_A.csv & ./data/test_B.csv), which correponds to Rank A and Rank B in evaluations. As a classification task, each user in the dataset has a unique id ('CUST_UID') and a 50-dimensional feature, as well as a label  ('LABEL') indicating whether the user is churning or not. The model's evaluation metric is AUC.

Here I present my LightGBM-based solution in this contest, which follows the steps to build a ML model: pre-processing, feature engineering and model fusion. My solution achieves 0.9516 in Rank A and 0.7539 in Rank B, which still remains much place for improvement.
