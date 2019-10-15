# Next steps for the tentative data analysis

## Model

* [x] 6 param: fit for everyone
* [x] Identify plateau: in sigma_obs and pi_lapse
    * [x] 12 colormaps with 2 dim.
    * [x] Choose sigma_obs and pi_lapse for the 4 param model
* [x] Fit 4 param model (with Bayesian pi_lapse)
* [x] Latex doc: "From exp-distr. prior over pi_lapse to an effective single value"
* [ ] sigma_obs vs sigma_stim colormap: justify the choice sigma_stim = 0 (three repr. participants in exp 1)
    * [ ] Conclusion: 4 instead of 7 parameters
* [x] Fit a meta-human observer directly on the aggregate data of exp 1

## Experiment 1

* [ ] Split into training and test set (not along the repetition lines): How good can we predict the confusion matrix then?
* [x] How do the confusion matrices, the predictive power and the LL look when the meta-human model is used?
* [x] Shuffle human responses and fit again: the model can not fit random effects!
    * [x] Shuffle "globally"
    * [x] Also: only within gt-structure (to test "fine structure of model") ["If the log-likelihood goes down on the shuffled data, then the model actually predicted per-trial."]

## Experiment 2

* [ ] x-axis: glo factor; y-axis: choice (CLU vs SDH)
* [ ] x-axis: log-likelihood difference (w/ or w/o obs noise); y-axis: choice (CLU vs SDH)
    * [ ] logistic fct fit I: aggregate human choices (all confidences)
    * [ ] logistic fct fit II: aggregate model predicted choice probabilities
* [ ] Fit on exp 1 with only temperature as free parameter (reduced model with correct flat prior): Is this enough to explain the shallow response prediction?
* [ ] How do the per-participant prediction curves look when the meta-human model is used (maybe evaluate as log-likelihood)?

## Confidence 

* [ ] Run discriminability with sigma_obs > 0 and = 0: and then logistic regression on abs(Delta) OR Delta^2
* [ ] Sanders style evaluation of correct and false answers as fnct of Delta
* [ ] Fit a confidence model as p(conf) = sigma(p(struct=human choice); slope, bias)
  * [ ] Is the pridiction better than just the prior (avg number of high-conf choices)
  * [ ] This model could also be used for preditions in exp 2.
* [ ] Fit the Gauss-confidence curve to exp 2 based on (signed) LL-difference (3 params: mean, variance, height)
    * [ ] With the exp-1-fitted confidence model, we can generate model data points (like human data) and do another Gaussian fit.
  
## Learning and consistency 

* [ ] Learning:
    * [ ] split exp 1 data in 1st and 2nd repetition: points; confidence; confusion matrices [aggregate data]
    * [ ] Same for 1st 100 vs 2nd 100 trials
* [x] Consistency and prediction power:
    * [x] _How often are the two answers of the repetitions identical?_ --> Best possible predictability (noise ceiling)
    * [x] Chance level: 1/4 --> _How well does the model predict human responses?_ p(choice=human answer)
    * [x] Future work: Compare impact of different model components on prediction power

## Computational role of factors in the observer model

* [x] Measures:
    * [ ] Example confusion matrices (for below models for one representative participant) [good for visual]
    * [x] Prediction power (bar plot) [good for intuitive number]
    * [x] Log-likelihood [actual objective function]
* [x] Models (all max likelihood fits): 
    1. Full model: 6 parameter model
    2. Reduced 4 parameter model
    3. Remove one of the factors alltogether: "full model minus X" with X=sigma_obs, pi_lapse, temp, prior (4 sub-models; remove = set to zero)
    4. Ideal observer (with obs noise)
    5. Ideal observer (without obs noise)
