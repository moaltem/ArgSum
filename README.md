# Argument Summarization (ArgSum) - Generating and Evaluating Summaries of Argumentative Discourse with LLMs

This GitHub repository contains the data, models, and code used in the investigations for my master's thesis.

**Preparations**: 

1. Replace the *models* folder with the following folder from Google Drive: https://drive.google.com/drive/folders/1GUzNhU6DK3KRUV-f4cX2xEb8ifTJKhm6
2. Insert your username and password of the Summetix API service into *argsum/___summetix_login.json*

**Structure**: 

- *data* folder: Datasets
- *models* folder: Language models (LMs) (divided into Match Scorers, Quality Scorers, Metics, and ArgSum Generators)
- *argsum* folder: Python code for functions and classes used in the investigations (+ the code for BLEURT and a json including the login information for the Summetix API service)
- *investigations* folder: Data resulting from the investigations
- Jupyter notebooks: Conducted investigations and results

**Investigations (.ipynb)**:

1. *data_processing*: Preparation of the raw data for the investigations
2. *explorative_data_analysis*: Exploratory data analysis
3. *quality_scorer*: Fine-tuning of LMs for argument quality scoring (+ their evaluation)
4. *match_scorer*: Fine-tuning of LMs for determining a match score between an argument and argument summary (+ their evaluation)
5. *flan_t5_sum*: Fine-tuning of FLAN T5 for argument summary generation (given a cluster of similar arguments)
6. *human_eval*: Examination of inter-rater reliability and the correlation between human judgements and automatic evaluation metrics
7. *arg_seperation_capability*: Examination of the ability of clustering-based ArgSum systems to separate arguments 
8. *get_cluster_sums*: Generate argument summaries with clustering-based ArgSum systems
9. *get_classification_sums*: Generate argument summaries with classification-based ArgSum systems
10. *eval_sums*: Automatic evaluation of the generated argument summaries
