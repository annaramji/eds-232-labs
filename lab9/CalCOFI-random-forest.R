

set.seed(1) # for pseudo-reproducibility
# create recipe
# remove some variables after looking at variable importance plot
recipe <- recipe(dic ~ #`lat_dec` +
                   #`lon_dec` +
                   #`r_temp` + # reported temp of air or water???
                   `r_depth` + # reported depth
                   `r_sal` + # Reported Salinity (from Specific Volume Anomoly, M³/Kg) (density)
                   `ta1_x` + # total alkalinity
                   `no2u_m` + # nitrite
                   `no3u_m` + # nitrate
                   #`nh3u_m` + # ammonia
                   `salinity1` + # Salinity (Practical Salinity Scale 1978) (conductivity)
                   `temperature_deg_c` + # water temp
                   `r_nuts` + # ammonium
                   `r_oxy_micromol_kg` + # oxygen
                   `r_dynht` + # Reported Dynamic Height in units of dynamic meters (work per unit mass)
                   `po4u_m`+ # phosphate
                   `si_o3u_m`, # silicate
                 data = train) %>%
  step_zv(all_predictors()) %>% # remove variables that only have a single value
  step_normalize(all_numeric_predictors())

# specify model
# tune all three parameters
rf_model <- rand_forest(mtry = tune(),
                        trees = tune(),
                        min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# create workflow
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

# cross validation
cv_fold <- vfold_cv(train, v = 10)

doParallel::registerDoParallel(cores = 4)

# define cross-validation for tuning
system.time(
  rf_cv_tune <- rf_workflow %>%
    tune_grid(resamples = cv_fold,
              grid = 10)
)

# finalize workflow with tuned parameters
rf_final_wf <- finalize_workflow(rf_workflow, select_best(rf_cv_tune, metric = "rmse"))

# fit on train
rf_fit <- fit(rf_final_wf, train)

# predict on full_test
final_results <- predict(rf_fit, test) %>%
  bind_cols(test)

# df for final submission
rf_final <- final_results %>%
  select(id, .pred) %>%
  rename(DIC = .pred)


