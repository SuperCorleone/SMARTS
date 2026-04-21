---  
**Data Source and Splits**  

The raw data comes from two CSV files: a training set (approx. 450 records) and a validation set (462 records). Each record contains 9 sensor features and a binary label, `hazard`.  

We use a single deployment scenario (Setting A): the entire training set (450 records) is used as warm-up data to train the model. The validation set (462 records) is then treated as an incoming data stream for online evaluation, one record at a time.  

---  
**Model Training Process**  

**Step 1: Build candidate model library**  

After obtaining the warm-up data, the system decides which feature combinations to model. It starts with single features and gradually tries combinations of two, three, up to 15 features. However, it does not enumerate blindly—at each round, only statistically adequate models are kept (using the BIC criterion, discarding models whose likelihood is less than 1/20 of the best model’s). The next round only expands on the surviving models. This process is called *Occam’s Window* pruning, ultimately yielding dozens to hundreds of "worthwhile" feature subsets, each corresponding to a logistic regression model.  

**Step 2: Train the stacking meta-learner**  

With the base models ready, 10-fold cross-validation is used to generate predicted probabilities from each base model for each sample. These probabilities are transformed to log-odds space (i.e., applying the logit transformation), resulting in a matrix where rows are samples and columns are log-odds predictions from each base model. A stochastic gradient descent (SGD) linear classifier (learning rate 0.001, L2 regularization) is then trained to learn how to "weighted combine" the opinions of the base models into a final prediction. Finally, all base models are retrained on the full warm-up data to obtain the complete stacking model.  

The key design here is: the base model ensemble (dozens to hundreds of logistic regressions) is responsible for "viewing the data from different perspectives," while the meta-learner is responsible for "integrating the opinions for the final decision."  

**Step 3: Train baseline methods**  

For comparison, the same warm-up data is also used to train BMA (Bayesian Model Averaging) and Best-Logit (the single highest-F1 model among the stacking base models). Additionally, an Oracle BMA is trained on the validation set, representing an "ideal predictor with future information," serving as the evaluation benchmark.  

---  
**Online Prediction and Update (Prequential Evaluation)**  

After model training, the data stream arrives one record at a time. For each new data point, the procedure is *predict first, then update*—strictly following the MAPE-K loop semantics, ensuring that at prediction time, the model has not yet seen the true label of that record.  

- **Prediction phase**: The new sample is fed into each base model, each outputs a probability, which is converted to log-odds. These log-odds are assembled into a feature vector and passed to the meta-learner for the final decision.  

- **Update phase (Stacking-Online only)**: The new sample’s meta-features and true label are added to a mini-batch buffer. After every 5 samples, a partial fit (stochastic gradient descent weight update) is performed on the meta-learner. Concurrently, a drift detection window is maintained: the prediction errors of the most recent 200 samples are split into two halves. If the average error of the second half exceeds that of the first half by more than the drift threshold (swept in parallel over values such as {0.05, 0.10, 0.15, 0.20, 0.25}), concept drift is declared. When drift fires, the meta-learner is warm-restarted: it performs 3 randomised passes of `partial_fit` over the retained retrain buffer (capped at 150 samples) while preserving its existing weights, so it adapts toward the new distribution without discarding prior knowledge. There is no minimum-buffer gate — any non-empty buffer is eligible. Cold restart is not used because it inflated variance and loss in our preliminary runs.  

It is important to note that online updates modify only the meta-learner’s weights; all base models remain unchanged from their warm-up trained state. This means saving the online model state is very lightweight—only the coefficient vector and intercept of the meta-learner need to be recorded.  

---  
**GA Adaptive Optimization**  

For samples in the data stream where `hazard = 0` (currently safe), the system initiates a genetic algorithm (GA) to search for the "most dangerous configuration." Specifically, the GA searches within the allowed ranges of four robot-controllable parameters (power, frequency band, mass, speed) to find a set of parameter values that maximize the model’s predicted hazard probability while minimizing the cost of parameter changes. The change cost for each parameter is weighted by its importance (power weight = 0.8, highest; speed weight = 0.1, lowest). The GA runs for 50 generations with a population of 100, finally outputting a hazard prediction for the optimal configuration.  

Then, the Oracle BMA (trained on the validation set as the ideal model) also makes a prediction for the same optimal configuration, serving as the "ground truth." The comparison yields two metrics: RE (relative error), measuring prediction deviation, and Success (GA prediction > 0.5 and Oracle prediction > 0.5), measuring whether the direction is correctly identified.  

---  
Note:

- **RQ1**: Stacking-Online is the only model that performs continual learning; all other models remain frozen in their warm-up trained state.  

- **RQ2**: The distinctive aspect of Stacking-Online is that at different time points in the data stream, it uses different model states (due to online updates). Therefore, for each `hazard = 0` moment, the model snapshot at that specific time must be restored to run the GA.