/* Final project */
ODS HTML;
ODS LISTING CLOSE;
ODS GRAPHICS ON;


/*Creating new dataset as "advertise" and reading the data*/
LIBNAME Proj "Proj";
data advertise;
set Proj.DATA;
run;

/* Create training and test datasets. 70% of sample in training  */
proc surveyselect data=advertise out=advertise_sampled outall samprate=0.7 seed=10;
run;

data ad_training ad_test;
set advertise_sampled;
if selected then output ad_training; 
else output ad_test;
run;


/* 1. Linear probability model*/

/* Generate indicator variables using glm_mod to run proc reg for the categorical variables */
proc glmmod data=advertise_sampled outdesign=LPM_with_indicators noprint; 
class device_platform_class;
model install = device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class/noint;
weight selected;
run;

/* Generate indicator variables using glm_mod for training dataset */
proc glmmod data=ad_training outdesign=ad_training_with_indicators noprint; 
class device_platform_class;
model install = device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class/noint;
run;

 /* Generate indicator variables using glm_mod for test dataset */
proc glmmod data=ad_test outdesign=ad_test_with_indicators noprint; 
class device_platform_class;
model install = device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class/noint;
run;

proc contents data= ad_training_with_indicators; 
run;

/* Initial Model: Linear probability model using PROC reg */
proc reg data=ad_training_with_indicators  PLOTS(MAXPOINTS=100000);
 model install = col1 - col10 ;
quit; 


/* Trial 1*/
/* Log model - not chosen */
data advertiselog;
set Proj.DATA;
log_device_volume = log(device_volume);
log_resolution = log(resolution);
ldevice_height = log(device_height);
log_dev_width = log(device_width);
run;

proc surveyselect data=advertiselog out=ad_sampledlog outall samprate=0.7 seed=10;
run;

data ad_traininglog ad_testlog;
set ad_sampledlog;
if selected then output ad_traininglog; 
else output ad_testlog;
run;

proc glmmod data=ad_sampledlog outdesign=ad_outputlog noprint;
class device_platform_class;
model install = publisher_id_class device_make_class device_platform_class device_os_class device_height device_width resolution device_volume wifi
log_dev_width log_device_volume log_resolution ldevice_height;
weight selected;
run;
proc contents data=ad_outputlog;
run;
                                                                                                                                                                                                                                           
proc reg data=ad_outputlog;
Withoutlog : model install = col2 - col11;
Logmodel : model install = col2- col6 col11-col15;
weight selected;
output out=ad_problog pred=p;
quit;
/* Log model - not chosen */



/* ITERATIVE APPROACHES */
/* Trial 2 - Forward Selection*/
proc glmselect data=ad_training_with_indicators plots=all;
model install = col1- col10/selection=forward(select=sl sle=0.2) stats=all showpvalues;
run;

/* Trial 3 - Backward Selection*/
proc glmselect data=ad_training_with_indicators plots=all;
model install = col1- col10/selection=backward(select=sl sls=0.15) stats=all showpvalues;
run;


/* Trial 4- Stepwise Selection*/
proc glmselect data=ad_training_with_indicators plots=all;
model install = col1- col10/selection=stepwise(select=sl) stats=all showpvalues;
run;

/* Trial 5- Best subsets regression */
proc reg data=ad_training_with_indicators plots=none;
 model install = col1- col10 /selection=cp adjrsq aic bic best=10;
quit;
proc contents data=ad_training_with_indicators ;
run;


/* Using ASE in train vs. test data  */
/* Creating a smaller version of the dataset */
data smallsample; 
set ad_training_with_indicators; 
call streaminit(1);
if rand('uniform') > 0.75;
run;

/* Trial 6 - Backward selection(ASE in train vs. test data) with p-value as criteria to see if adding interaction terms reduce ASE values*/
proc glmselect data=smallsample testdata=ad_test_with_indicators seed=10 plots=ase;
model install = col1 | col2 | col3 | col4 | col5 | col6 | col7 | col8 @2
/selection=backward(select=sl sls=0.15) hierarchy=single;
performance buildsscp=incremental;
run;


/* Trial 7 - Backward selection with AIC as criteria(ASE in train vs. test data) to see if adding interaction terms reduce ASE values*/
proc glmselect data=smallsample testdata=ad_test_with_indicators seed=10 plots=ase;
model install = col1 | col2 | col3 | col4 | col5 | col6 | col7 | col8 @2
/selection=backward(select=aic) hierarchy=single;
performance buildsscp=incremental;
run;


/* Use of cross-validation to choose best model*/
/* Trial 8 - Backward selection with p-value as criteria and validation dataset to choose predictors*/
proc glmselect data=smallsample testdata=ad_test_with_indicators seed=10 plots=all;
partition fraction(validate=0.2); 
model install = col1 | col2 | col3 | col4 | col5 | col6 | col7 | col8 @2
/selection = backward(select=sl choose=validate);
run;


/* Trial 9 - Backward selection with AIC as criteria and validation dataset to choose predictors*/
proc glmselect data=smallsample testdata=ad_test_with_indicators seed=10 plots=all;
partition fraction(validate=0.2); 
model install = col1 | col2 | col3 | col4 | col5 | col6 | col7 | col8 @2
/selection = backward(select=AIC choose=validate);
run;


/* Final Model: Linear probability model using PROC reg */
proc reg data=ad_training_with_indicators plots(maxpoints=100000);
Final_Model : model install = col1 - col8  ;
output out=ad_lin_predict_final p=linear_predictions;
quit;



/* Logistic Regression */
/* Initial model - Logistic Regression */
proc logistic data=ad_training_with_indicators;
 logit: model install (event='1') = col1 - col10; 
run;


/* Trial 1 - Stepwise Selection*/
proc logistic data=ad_training_with_indicators outest=betas covout;
logit: model install (event='1') = col1 - col10
/ selection=stepwise
                  slentry=0.25
                  slstay=0.35
                  details
                  lackfit;
run;


/* Trial 2 - Forward Selection*/
proc logistic data=ad_training_with_indicators outest=betas1 covout;
logit: model install (event='1') = col1 - col10
/ selection=forward
                  slentry=0.25
                  details
                  lackfit;
run;


/* Trial 3 - Backward Selection*/
proc logistic data=ad_training_with_indicators outest=betas2 covout;
logit: model install (event='1') =   col1 - col10 
/ selection=backward
                  slstay=0.35
                  details
                  lackfit;
run;



/* Final model - Logistic Regression */
/* (i) Estimation of the model without considering rare events*/
proc logistic data=ad_training_with_indicators;
logit: model install (event='1') = col1 - col8; 
run;

/*(ii)b. Estimate the model using oversampling approach for handling rare events and then applying the correction to obtain the corrected intercept */
proc freq data=advertise_sampled;
table install / out=fullpct(where=(install=1) rename=(percent=fullpct));
title "response counts in full data set";
run;

data sub;
set advertise_sampled;
if install =1 or (install =0 and ranuni(75302)<1/119) then output;
run;

proc freq data=sub;
table install / out=subpct(where=(install =1) rename=(percent=subpct));
title "Response counts in oversampled, subset data set";
run;

data sub;
set sub;
if _n_=1 then set fullpct(keep=fullpct);
if _n_=1 then set subpct(keep=subpct);
p1=fullpct/100; r1=subpct/100;
w=p1/r1; if install =0 then w=(1-p1)/(1-r1);
off=log( (r1*(1-p1)) / ((1-r1)*p1) );
run;

proc logistic data=sub;
class device_platform_class;
model install (event="1")=device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class;
output out=out p=pnowt;
title "True Parameters: -8.1248 (intercept)";
title2 "Unadjusted Model";
run;

proc logistic data=out;
class device_platform_class;
model install (event="1")=device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class;
weight w;
output out=out p=pwt;
title2 "Weight-adjusted Model";
run;

proc logistic data=out;
class device_platform_class;
model install (event="1")=device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class / offset=off;
output out=out xbeta=xboff;
title2 "Offset-adjusted Model";
run;



/* Classification and ROC */

/* Initial Model - Linear probability model */
/* Making predictions for test observations */
proc reg data=LPM_with_indicators  PLOTS(MAXPOINTS=100000);
linear: model install = col1 - col10 ;
output out=ad_lin_predict_initial p=linear_predictions;
weight selected; 
quit;

/* To plot ROC curve based on predictions from linear model */
proc logistic data=ad_lin_predict_initial  plots=roc(id=prob);
model install (event='1') = col1 - col10/ nofit;
roc pred=linear_predictions;
where selected=0;
run;


/* Final Model: Linear probability model using PROC reg */
/* Making predictions for test observations */
proc reg data=LPM_with_indicators  PLOTS(MAXPOINTS=100000);
linear: model install = col1 - col8 ;
output out=ad_lin_predict_final p=linear_predictions;
weight selected; 
quit;

/* To plot ROC curve based on predictions from linear model */
proc logistic data=ad_lin_predict_final  plots=roc(id=prob);
model install (event='1') = col1 - col8/ nofit;
roc pred=linear_predictions;
where selected=0;
run;



/*Initial Model- Logistic regression */
/* Make predictions on test data */
proc logistic data=ad_training_with_indicators;
logit: model install (event='1') = col1 - col10; 
score data=ad_test_with_indicators out=ad_logit_predict_initial; 
run;

/*ROC curve on test data */
proc logistic data=ad_logit_predict_initial plots=roc(id=prob);
model install (event='1') = col1 - col10/ nofit;
roc pred=p_1;
run;



/* Final model - Logistic Regression */
/* Estimation of the model without considering rare events*/
proc logistic data=ad_training_with_indicators;
logit: model install (event='1') = col1 - col8; 
score data=ad_test_with_indicators out=ad_logit_predict_final; 
run;

/*ROC curve on test data */
proc logistic data=ad_logit_predict_final plots=roc(id=prob);
model install (event='1') = col1 - col8/ nofit;
roc pred=p_1;
run;


/* PART 2 */
/* (i) ROC Table for Initial Logistic Regression Model */
proc logistic data=ad_training_with_indicators outmodel=ad_training_roc1;
logit: model install (event='1') = col1 - col10; 
run;
proc logistic inmodel=ad_training_roc1;
score data=ad_test_with_indicators outroc=adlog_roc1;
run;
/* Finding total cost for each probability threshold based on misclassification */
data ad_threshold1;
set adlog_roc1;
total_cost1 = _FALPOS_*0.01 + _FALNEG_*1;
run;
proc sql; 
create table total_cost1a as(select*,min(total_cost1) as min_cost from ad_threshold1); 
create table total_cost1b as select _PROB_ as corresponding_prob from ad_threshold1 where total_cost1=281.5;
run;

/* (ii) ROC Table for Final Logistic Regression Model */
proc logistic data=ad_training_with_indicators outmodel=ad_training_roc2;
logit: model install (event='1') = col1 - col8; 
run;
proc logistic inmodel=ad_training_roc2;
score data=ad_test_with_indicators outroc=adlog_roc2;
run;
/* Finding total cost for each probability threshold based on misclassification */
data ad_threshold2;
set adlog_roc2;
total_cost2 = _FALPOS_*0.01 + _FALNEG_*1;
run;
proc sql; 
create table total_cost2a as(select*,min(total_cost2) as min_cost from ad_threshold2); 
create table total_cost2b as select _PROB_ as corresponding_prob from ad_threshold2 where total_cost2=282.11;
run;


/* (ii) ROC Table for Initial Linear Probability Model */
data temp_a (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.001 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_a as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_a;
quit;


data temp_b (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.005 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_b as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_b;
quit;

data temp_c (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.010 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_c as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_c;
quit;


data temp_d (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.015 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_d as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_d;
quit;


data temp_e (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.020 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_e as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_e;
quit;


data temp_f (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.025 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_f as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_f;
quit;


data temp_g (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.030 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_g as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_g;
quit;


data temp_h (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.035 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_h as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_h;
quit;


data temp_i (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.040 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_i as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_i;
quit;


data temp_j (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.045 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_j as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_j;
quit;


data temp_k (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.050 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_k as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_k;
quit;


data initial_linear_roc_temp;
input probability false_positive false_negative;
datalines;
0.001 36061 0
0.005 32438 0
0.010 7611 0
0.015 125 0
0.020 0 0
0.025 0 0
0.030 0 0
0.035 0 0
0.040 0 0
0.045 0 0
0.050 0 0 
;
run;

data initial_linear_roc;
set initial_linear_roc_temp;
total_cost = false_positive*0.01 + false_negative*1;
run;


/* (ii) ROC Table for Final Linear Probability Model */

data temp_a1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_final;
where selected=0;
if linear_predictions > 0.001 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_a1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_a1;
quit;


data temp_b1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.005 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_b1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_b1;
quit;

data temp_c1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.010 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_c1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_c1;
quit;


data temp_d1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.015 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_d1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_d1;
quit;


data temp_e1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.020 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_e1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_e1;
quit;


data temp_f1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.025 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_f1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_f1;
quit;


data temp_g1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.030 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_g1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_g1;
quit;


data temp_h1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.035 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_h1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_h1;
quit;


data temp_i1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.040 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_i1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_i1;
quit;


data temp_j1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.045 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_j1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_j1;
quit;


data temp_k1 (keep=install selected linear_predictions predicted false_pos false_neg);
set ad_lin_predict_initial;
where selected=0;
if linear_predictions > 0.050 then predicted=1;
if install=0 and predicted=1 then false_pos=1;
if install=1 and predicted=0 then false_neg=1;
run;

proc sql;
create table count_k1 as select*,count(false_pos) as count_fp, count(false_neg) as count_fn from temp_k1;
quit;


data final_linear_roc_temp;
input probability false_positive false_negative;
datalines;
0.001 36067 0
0.005 32438 0
0.010 7611 0
0.015 125 0
0.020 0 0
0.025 0 0
0.030 0 0
0.035 0 0
0.040 0 0
0.045 0 0
0.050 0 0 
;
run;

data final_linear_roc;
set final_linear_roc_temp;
total_cost = false_positive*0.01 + false_negative*1;
run;



