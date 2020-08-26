# h2oclasseval
Experimental R package for evaluating classification models built with h2o automl

## Install

```
library(devtools)

install_github("BillPetti/h2oclasseval')

```

## Prep Demo Data from the Titanic Data set

We will set the outcome to classify as `Class` from `Survived`:

```
library(h2o)
library(h2oclasseval)
library(tidyverse)
library(titanic)

titanic <- titanic::titanic_train

titanic <- titanic %>%
  select(-c(PassengerId, Name, Ticket, Cabin)) %>%
  mutate(Survived = factor(Survived, levels = c(0,1)), 
         Pclass = factor(Pclass, levels = c(1,2,3)), 
         Sex = as.factor(Sex), 
         Embarked = as.factor(Embarked)) %>%
  rename(Class = Survived) %>%
  mutate_if(is.numeric, scale) %>%
  mutate_if(is.matrix, as.numeric)

train <- titanic %>%
  sample_frac(.75)

test <- dplyr::setdiff(titanic, train)
```

## Fit Classifier Models for Passenger Survival with `h2o::automl`

For speed, we'll exclude any Deep Learning algorithms:

```
if(tryCatch({
  h2o.clusterIsUp()
}, error=function(e) "No h2o Instance Running") != "No h2o Instance Running") {
  h2o.shutdown(prompt = F)
}

h2o.init(max_mem_size = "8g")

y <- "Class"
x <- setdiff(names(train), y)

h2o_train <- as.h2o(train)
h2o_test <- as.h2o(test)

# fit models -----

aml <- h2o.automl(x = x,
                  y = y,
                  training_frame = h2o_train,
                  leaderboard_frame = h2o_train,
                  max_models = 60,
                  include_algos = c("DRF", "GLM", "GBM"), 
                  #exclude_algos = c("DeepLearning"),
                  max_runtime_secs = 3600)

lb <- aml@leaderboard
lb_dataframe <- aml@leaderboard %>%
  as.data.frame()
```

## Saving the model objects

Set a `path_slug` to an existing director and a `time_stamp` if different from function's default:
```
path_slug <- '/Users/williampetti/Desktop/h2o_titanic/'
time_stamp <- gsub('-| |:', '_', Sys.time())

h2o_payload <- save_h2o_models_varimport(leaderboard = aml@leaderboard,
                                         path_slug = path_slug, 
                                         time_stamp = time_stamp)
```

You should now see individual h2o models and a object that collects each object and their variable importance (if applicable):

```
> list.files(path_slug)
 [1] "2020_08_25_11_29_11_h2o_train_model_metrics.rds"
 [2] "DRF_1_AutoML_20200825_112853"                   
 [3] "GBM_1_AutoML_20200825_112853"                   
 [4] "GBM_2_AutoML_20200825_112853"                   
 [5] "GBM_3_AutoML_20200825_112853"                   
 [6] "GBM_4_AutoML_20200825_112853"                   
 [7] "GBM_5_AutoML_20200825_112853"                   
 [8] "GBM_grid__1_AutoML_20200825_112853_model_1"     
 [9] "GBM_grid__1_AutoML_20200825_112853_model_2"     
[10] "GBM_grid__1_AutoML_20200825_112853_model_3"     
[11] "GBM_grid__1_AutoML_20200825_112853_model_4"     
[12] "GBM_grid__1_AutoML_20200825_112853_model_5"     
[13] "GBM_grid__1_AutoML_20200825_112853_model_6"     
[14] "GBM_grid__1_AutoML_20200825_112853_model_7"     
[15] "GBM_grid__1_AutoML_20200825_112853_model_8"     
[16] "GLM_1_AutoML_20200825_112853"                   
[17] "XRT_1_AutoML_20200825_112853"     
```
Variable importance for each model can be accessed like so:

```
> h2o_payload$DRF_1_AutoML_20200825_112853$variable_importance
                      model_id variable relative_importance scaled_importance percentage
1 DRF_1_AutoML_20200825_112853      Sex           1109.5988         1.0000000 0.32190429
2 DRF_1_AutoML_20200825_112853      Age            732.1392         0.6598234 0.21239998
3 DRF_1_AutoML_20200825_112853     Fare            728.6841         0.6567095 0.21139762
4 DRF_1_AutoML_20200825_112853   Pclass            426.1224         0.3840329 0.12362184
5 DRF_1_AutoML_20200825_112853    SibSp            167.3491         0.1508195 0.04854945
6 DRF_1_AutoML_20200825_112853 Embarked            157.0859         0.1415700 0.04557200
7 DRF_1_AutoML_20200825_112853    Parch            126.0039         0.1135581 0.03655483
```

You can also generate a custom grid with a number of evaluation metrics for classification models. The grid will calculate overall `logloss` as well as custom metrics for individual thresholds:
```
# pull model list

model_list <- map(h2o_payload, ~.$model_object)

# generate custom metrics with different thresholds

h2o_test_metrics <- map_df(.x = model_list,
                           ~custom_h2o_confusion_metrics(model = .x,
                                                         newdata = h2o_test,
                                                         threshold = c(0,1)))
                                                         
# generate table for models with the lowest logloss and precisin and recall at or above 75%

h2o_test_metrics %>%
  filter(precision >= .75, 
         recall >= .75) %>%
  arrange(logloss, desc(precision), desc(recall), desc(f1_score)) %>%
  slice(1:10) %>%
  bpettir::format_tables_md()
```                                                         

<table class="table table-striped table-hover table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> model </th>
   <th style="text-align:right;"> threshold </th>
   <th style="text-align:right;"> logloss </th>
   <th style="text-align:right;"> precision </th>
   <th style="text-align:right;"> recall </th>
   <th style="text-align:right;"> neg_precision </th>
   <th style="text-align:right;"> specificity </th>
   <th style="text-align:right;"> fall_out </th>
   <th style="text-align:right;"> f1_score </th>
   <th style="text-align:right;"> true_positives </th>
   <th style="text-align:right;"> false_negatives </th>
   <th style="text-align:right;"> false_positives </th>
   <th style="text-align:right;"> true_negatives </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_3 </td>
   <td style="text-align:right;"> 0.42 </td>
   <td style="text-align:right;"> 0.4529051 </td>
   <td style="text-align:right;"> 0.7500000 </td>
   <td style="text-align:right;"> 0.7702703 </td>
   <td style="text-align:right;"> 0.8380952 </td>
   <td style="text-align:right;"> 0.8224299 </td>
   <td style="text-align:right;"> 0.1775701 </td>
   <td style="text-align:right;"> 0.7600000 </td>
   <td style="text-align:right;"> 57 </td>
   <td style="text-align:right;"> 17 </td>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 88 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_1 </td>
   <td style="text-align:right;"> 0.51 </td>
   <td style="text-align:right;"> 0.4539179 </td>
   <td style="text-align:right;"> 0.7777778 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8348624 </td>
   <td style="text-align:right;"> 0.8504673 </td>
   <td style="text-align:right;"> 0.1495327 </td>
   <td style="text-align:right;"> 0.7671233 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 16 </td>
   <td style="text-align:right;"> 91 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_1 </td>
   <td style="text-align:right;"> 0.52 </td>
   <td style="text-align:right;"> 0.4539179 </td>
   <td style="text-align:right;"> 0.7777778 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8348624 </td>
   <td style="text-align:right;"> 0.8504673 </td>
   <td style="text-align:right;"> 0.1495327 </td>
   <td style="text-align:right;"> 0.7671233 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 16 </td>
   <td style="text-align:right;"> 91 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_1 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> 0.4539179 </td>
   <td style="text-align:right;"> 0.7671233 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8333333 </td>
   <td style="text-align:right;"> 0.8411215 </td>
   <td style="text-align:right;"> 0.1588785 </td>
   <td style="text-align:right;"> 0.7619048 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 17 </td>
   <td style="text-align:right;"> 90 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_1 </td>
   <td style="text-align:right;"> 0.49 </td>
   <td style="text-align:right;"> 0.4539179 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8317757 </td>
   <td style="text-align:right;"> 0.8317757 </td>
   <td style="text-align:right;"> 0.1682243 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 89 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_6 </td>
   <td style="text-align:right;"> 0.46 </td>
   <td style="text-align:right;"> 0.4540665 </td>
   <td style="text-align:right;"> 0.7532468 </td>
   <td style="text-align:right;"> 0.7837838 </td>
   <td style="text-align:right;"> 0.8461538 </td>
   <td style="text-align:right;"> 0.8224299 </td>
   <td style="text-align:right;"> 0.1775701 </td>
   <td style="text-align:right;"> 0.7682119 </td>
   <td style="text-align:right;"> 58 </td>
   <td style="text-align:right;"> 16 </td>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 88 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_grid__1_AutoML_20200825_112853_model_6 </td>
   <td style="text-align:right;"> 0.47 </td>
   <td style="text-align:right;"> 0.4540665 </td>
   <td style="text-align:right;"> 0.7500000 </td>
   <td style="text-align:right;"> 0.7702703 </td>
   <td style="text-align:right;"> 0.8380952 </td>
   <td style="text-align:right;"> 0.8224299 </td>
   <td style="text-align:right;"> 0.1775701 </td>
   <td style="text-align:right;"> 0.7600000 </td>
   <td style="text-align:right;"> 57 </td>
   <td style="text-align:right;"> 17 </td>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 88 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_2_AutoML_20200825_112853 </td>
   <td style="text-align:right;"> 0.55 </td>
   <td style="text-align:right;"> 0.4644109 </td>
   <td style="text-align:right;"> 0.8000000 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8378378 </td>
   <td style="text-align:right;"> 0.8691589 </td>
   <td style="text-align:right;"> 0.1308411 </td>
   <td style="text-align:right;"> 0.7777778 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 93 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_2_AutoML_20200825_112853 </td>
   <td style="text-align:right;"> 0.56 </td>
   <td style="text-align:right;"> 0.4644109 </td>
   <td style="text-align:right;"> 0.8000000 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8378378 </td>
   <td style="text-align:right;"> 0.8691589 </td>
   <td style="text-align:right;"> 0.1308411 </td>
   <td style="text-align:right;"> 0.7777778 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 93 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> GBM_2_AutoML_20200825_112853 </td>
   <td style="text-align:right;"> 0.57 </td>
   <td style="text-align:right;"> 0.4644109 </td>
   <td style="text-align:right;"> 0.8000000 </td>
   <td style="text-align:right;"> 0.7567568 </td>
   <td style="text-align:right;"> 0.8378378 </td>
   <td style="text-align:right;"> 0.8691589 </td>
   <td style="text-align:right;"> 0.1308411 </td>
   <td style="text-align:right;"> 0.7777778 </td>
   <td style="text-align:right;"> 56 </td>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 93 </td>
  </tr>
</tbody>
</table>
