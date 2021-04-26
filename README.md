# Companies bankruptcy modelling with econometrics methods
Project created for *Advanced Econometrics I* (org. *Zaawansowana Ekonometria I*) classes at WNE UW

Language: 
 * Polish - classes, report and code comments

Semester: II (MA studies)

## About
The main objective of this project was to model Polish companies bankruptcy probability - binary classification task with statistical inference. The data was found on UCL machine learning repository and containes information about bankruptcy status and financial ratios in 5 task (differentiated by the forecasting horizon). In this project 1 year forecast horizon was picked. Data analysis was a big part of the project and contais such components as:
* handling missing values (categorical variable coded if there is a missing or not or observations deleting)
* feature engineering (some needed ratios were not present in the original dataset, fortunately it could be obtained by combining and transforming others),
* sample balance check
* outliers reduction (huge impact onmodels stability, deleting improper obsevrations based on expert analysis)
* mean equality between bankrupt and non bankrupt companies for each variable to gain some intuition about data and variable signigicances
* spearman correlation and scatter plots to detect redundant variables and prevent incomplete rank of the observation matrix (model may be unstable if not carried out)
* Variance Inflation Factor to detect redundant variables too

After analysis discrete choice models were considered (logistic regression, probit regression and linear probability model (project requirement)). Probit was selected for further analysis based on information criteria and statistical tests (F test). Next General to Specyfic procedure was performed (project requirement). Next step was to perform model diagnostics - functional form - linktest what causes creating new variables. After that godness of fit was checked using counted R^2, adjusted counted R^2 and R^2 McKelvey&Zavoina.

Research goals and hypotesis were as follows:
1. Try to develop a good predictive model for polish market ($P(y=0|\hat{y}=0)>0.95 and P(y)$)
