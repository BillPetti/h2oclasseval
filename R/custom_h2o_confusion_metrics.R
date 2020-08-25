#' Custom classification metrics for h2o automl models
#'
#' @param model An existin h2o model object.
#' @param newdata An h2o data object to apply the model object to. Typically a test set.
#' @param threshold A range of threshold/cut-off values to evaluate.
#'
#' @import h2o purrr dplyr tibble
#' @importFrom MLmetrics LogLoss
#' @return A data.frame containing metrics for each threshold value provided.
#'
#' @description
#'
#' model: The model name
#' threshold: Threshold value
#' logloss: The logloss of the model (see \url{https://www.kaggle.com/dansbecker/what-is-log-loss} for more information)
#' precision: True Positives / Predicted Positives
#' recall: True Positives / All Positives
#' neg_precision: Negative Prevision, True Negatives / Predicted Negatives
#' specificity: True Negatives / All Negatives
#' fall_out: 1 - \code{specificity}, used to plot ROC curves
#' f1_score: F1 Score or Balanced Accuracy, the harmonic mean of \code{precision} and \code{recall}
#' true_positives: Number of True Positives
#' false_negatives: Number of False Negatives
#' false_positives: Number of False Positives
#' true_negatives: Number of True Negatives
#'
#' @export
#'
#' @examples \dontrun{custom_h2o_confusion_metrics(model = NA, newdata = NA, threshold = c(0,1)}

custom_h2o_confusion_metrics <- function(model = NULL,
                                         newdata = NULL,
                                         threshold = c(.4,.6)) {

  if(is.null(model) == TRUE) {

    stop('Please provide an h2o model object.')
  }

  if(is.null(newdata) == TRUE) {

    stop('Please provide an h2o data set.')
  }

  map_thresholds <- function(predictions,
                             newdata,
                             threshold) {

    predictions_aug <- predictions %>%
      as.data.frame() %>%
      mutate(label = factor(ifelse(p1 > threshold, "1", "0"))) %>%
      mutate(truth = ifelse(newdata$Class == 1, "1", "0")) %>%
      mutate(truth = as.factor(truth))

    alpha_confusion_matrix <- predictions_aug %>%
      dplyr::group_by(label, truth) %>%
      dplyr::count() %>%
      dplyr::ungroup()

    master_confusion_matrix <- tibble::tibble(label = as.character(c(0,0,1,1)),
                                              truth = as.character(c(0,1,0,1)))

    confusion_matrix <- master_confusion_matrix %>%
      dplyr::left_join(alpha_confusion_matrix, by = c("label", "truth")) %>%
      replace(is.na(.), 0)

    true_positive_pred_positive <- as.numeric(confusion_matrix[4,3])
    true_positive_pred_negative <- as.numeric(confusion_matrix[2,3])
    true_negative_pred_positive <- as.numeric(confusion_matrix[3,3])
    true_negative_pred_negative <- as.numeric(confusion_matrix[1,3])

    precision <- as.numeric(true_positive_pred_positive / (true_negative_pred_positive+true_positive_pred_positive))
    recall <- as.numeric(true_positive_pred_positive / (true_positive_pred_positive+true_positive_pred_negative))
    neg_precision <- as.numeric(true_negative_pred_negative / (true_negative_pred_negative+true_positive_pred_negative))
    specificity <- as.numeric(true_negative_pred_negative / (true_negative_pred_negative+true_negative_pred_positive))
    fall_out <- 1-specificity
    logloss <- MLmetrics::LogLoss(y_pred = predictions_aug$p1,
                                  y_true = as.numeric(as.character(predictions_aug$truth)))
    f1_score <- (2 * precision * recall) / (precision+recall)

    payload <- tibble::tibble(model = model@model_id,
                              threshold = threshold,
                              logloss = logloss,
                              precision = precision,
                              recall = recall,
                              neg_precision = neg_precision,
                              specificity = specificity,
                              fall_out = fall_out,
                              f1_score = f1_score,
                              true_positives = true_positive_pred_positive,
                              false_negatives = true_positive_pred_negative,
                              false_positives = true_negative_pred_positive,
                              true_negatives = true_negative_pred_negative)

    return(payload)
  }

  if(is.h2o(newdata) == FALSE) {

    new_data <- h2o::as.h2o(newdata)

  } else {

    new_data <- newdata
  }

  predictions <- h2o::h2o.predict(object = h2o::h2o.loadModel(paste0(path_slug, model@model_id)),
                                  newdata = new_data) %>%
    as.data.frame()

  newdata <- as.data.frame(newdata)

  mapped_metrics <- purrr::map_df(.x = seq(threshold[1], threshold[2], .01),
                                  ~map_thresholds(predictions = predictions,
                                                  newdata = newdata,
                                                  threshold = .x))

  mapped_metrics
}

