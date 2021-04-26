# Companies bankruptcy modelling with econometrics methods
Project created for *Advanced Econometrics I* (org. *Zaawansowana Ekonometria I*) classes at WNE UW

Language: 
 * Polish - classes, report and code comments

Semester: II (MA studies)

## About
The main objective of this project was to model Polish companies bankruptcy probability - binary classification task with statistical inference. The data was found on UCL machine learning repository and contains information about bankruptcy status and financial ratios in 5 task (differentiated by the forecasting horizon). In this project 1 year forecast horizon was choosen. Variables was selected based on literature review (10 papers in total). Data analysis was a big part of the project and contais such components as:
* handling missing values (categorical variable coded if there is a missing or not or observations deleting)
* feature engineering (some needed ratios were not present in the original dataset, fortunately it could be obtained by combining and transforming others)
* sample balance check
* outliers reduction (huge impact on models stability, deleting improper obsevrations based on the expert analysis)
* mean equality between bankrupt and non bankrupt companies for each variable to gain some intuition about data and variables significance
* spearman correlation matrix and scatter plots to detect redundant variables and prevent incomplete rank of the observation matrix (model may be unstable if not carried out)
* Variance Inflation Factor to detect redundant variables too

After analysis discrete choice models were considered (logistic regression, probit regression and linear probability model (project requirement)). Probit was selected for further analysis based on information criteria and statistical tests (F test). Next General to Specyfic procedure was performed (project requirement). Next step was to perform model diagnostics - functional form - linktest which showed the need to create new variables. After that godness of fit was checked using counted R^2, adjusted counted R^2 and R^2 McKelvey&Zavoina.

Research hypothesis were as follows:
1. It is possible to develop a good predictive model for polish market (P(y=0|y_hat=0)>0.95 and P(y=1|y_hat=1)>0.35)
2. Company has negative impact on its bankruptcy probability
3. Adding Altman Z-score variable to the model reduces economics misclassification cost

To verify the 1st hypothesis confusion matrces were calculated and cutoff was adjusted. ROC curves were drawed (in and out of sample) and its integrals were computed. To verify the 2nd hypothesis marginal effects were computed and interpreted. 3rd hypothesis was verified via bootstrap simulation.

Findings:
 * financial statements data are hard to deal with, lots of missings and outliers
 * developed model met the requirements (1st hypothesis)
 * company size has negative impact on its bankruptcy probability (Too big to fail slogan seems to be true)
 * adding Z-score to the model can help with economics misclassification costs of the model but it is dependend from reasercher preferences

In this project I've learnt a looot of about econometric and statistical analysis, discrete choice models and statistical inference. I also improved my Python programming and problem solving skills, I had to implement some statistical tests, diagnostics tools from scratch with Python functions which was a lot of fun. What is more I've learnt some LaTeX for reports/papers writing (no more Word-written assignments) and how to find, read and understand research papers in english (10 papers included in literature review). There were no software requirements, but I've choosen Python because I really like it and want to learn as much as possible for Data Science and more (we could use STATA for example what is easy path to perform econometrics modelling). I wrote in my report that further works will be focused developing much more complex models for this problem and I kept that promise (machine learning projects including Master Thesis). I really enjoyed this problem.

## Repository description
 * Jak skuteczne jest sprawozdanie finansowe w przewidywaniu bankructwa firmy Kod zrodlowy.pdf - source code in pdf format, Python
 * Jak skuteczne jest sprawozdanie finansowe w przewidywaniu bankructwa firmy Kod zrodlowy.py - source code in py script format, Python
 * Jak skuteczne jest sprawozdanie finansowe w przewidywaniu bankructwa firmy. Notatnik.ipynb - Jupyter Notebook with analysis, Python
 * Jak skuteczne jest sprawozdanie finansowe w przewidywaniu bankructwa firmy.pdf - paper-style project report in pdf format 
 * Raport z pętli bootstrapowej.pdf - bootstrap loop report in pdf format
 * bootstrap.pkl - pickled bootstrap results, ready to load into environment 

## Technologies
 * Python (numpy, pandas, matplotlib.pyplot, scipy, statsmodels, patsy)
 * Jupyter Notebook
 * LaTeX (overleaf cloud environment)

## Author
Maciej Odziemczyk

## Notes
This is my first really big analysis in Python and I wanted it to be part of my Master Thesis, so I've tried very hard this analysis to be as good as possible. When I spent more time at WNE I realized that it is not enough for me.
