Updates explained:

- We change our SNR from 1/50 multiplicative noise to a additive noise of S/N of approximately 25, or σ = 0.04. Updates in this can be found in 'data/data_synthesization.py'
- Additionally, when labeling our filenames (which is later used to determine associated stellar labels), log g and radius are saved by cutting off at a specific decimal point as opposed to rounding to the nearest hundredth decimal point.
- Deleted everything related to clean data generation since it takes up processing time uneccessarily

In this iteration, we proceed with our original "unmasked" version of the ANN, and the corresponding results can be found below.

Label       MAE             RMSE
p_teff      2408.919286     2578.975436
s_teff      881.638765      991.301175
p_logg      0.747804        0.771880
s_logg      0.232476        0.288816
p_radius    0.200903        0.249363
s_radius    0.193806        0.220497