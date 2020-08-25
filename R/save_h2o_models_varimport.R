#' Save h2o automl models and variable importance
#'
#' @param leaderboard A leaderboard object generated from \code{h2o::automl}.
#' @param max_top_models Number of models to save and generate variable importance for. If NA, all
#' models will be processed.
#' @param path_slug A file path where the models should be saved.
#' @param time_stamp If TRUE, a time stamp is constructed using \code{Sys.time}. User can also
#' provide their own time stamp as a string e.g. '2020_08_01'.
#'
#' @return Function will return a list with an element for each model containing the model object
#' and the variable importance of the model (if appropriate--not availabel for Ensemble and GLM). Individual
#' models will be saved in the directory provided by the \code{path_slug} argument, and a list with all model
#' objects and variable importane will also be saved to the \code{path_slug} directory.
#'
#' @import h2o purrr dplyr tibble
#'
#' @export
#'
#' @examples \dontrun{save_h2o_models_varimport(leaderboard, max_top_models = NA,
#' path_slug, time_stamp = TRUE)}

save_h2o_models_varimport <- function(leaderboard,
                                      max_top_models = NA,
                                      path_slug,
                                      time_stamp = TRUE) {

  if(dir.exists(path_slug) == FALSE) {

    stop('The path_slug provided is either incorrect or does not exist. Operation stopped.')
  }

  loop_save_models <- function(model_from_leaderboard,
                               path_slug) {

    model <- h2o::h2o.getModel(model_from_leaderboard)

    h2o::h2o.saveModel(object = model,
                       force=TRUE,
                       path = path_slug)
  }

  message(paste0('Saving models to ', path_slug))

  leaderboard$model_id %>%
    as.data.frame() %>%
    {if (!is.na(max_top_models)) dplyr::slice(.data = ., 1:max_top_models)
      else dplyr::slice(.data = ., 1:length(.$model_id))} %>%
    dplyr::pull(model_id) %>%
    purrr::map(~loop_save_models(., path = path_slug))

  performance_data <- function(model_from_leaderboard,
                               path_slug) {

    model <- h2o::h2o.getModel(model_from_leaderboard)

    var_import <- model@model$variable_importances %>%
      as.data.frame() %>%
      dplyr::mutate(model_id = model@model_id) %>%
      dplyr::select(model_id, everything())

    model_list <- list(model_object = model,
                       variable_importance = var_import)

    model_list
  }

  names_to_pull <- leaderboard$model_id %>%
    as.data.frame() %>%
    {if (!is.na(max_top_models)) slice(.data = ., 1:max_top_models)
      else slice(.data = ., 1:length(.$model_id))} %>%
    dplyr::pull(model_id)

  payload <- names_to_pull %>%
    purrr::map(~performance_data(model_from_leaderboard = .)) %>%
    setNames(nm = names_to_pull)

  file_time_stamp <- if (time_stamp == TRUE) {

    file_time_stamp <- gsub('-| |:', '_', Sys.time())

  } else {

    file_time_stamp <- time_stamp

  }

  message(paste0('Saving model and variable importance list to ', path_slug))

  saveRDS(payload, file = paste0(path_slug, file_time_stamp, "_h2o_train_model_metrics.rds"))

  return(payload)

}

