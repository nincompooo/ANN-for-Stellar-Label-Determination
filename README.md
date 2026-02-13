### Updates explained:

- Remade a CLEAN dataset by removing our noise injection in our generation stage and moving it into our ANN stage
    - This allows us to finetune the amount of noise our dataset has and it's impact on improving the accuracy of our ANN
- Prevously, our problems with the way radius was being returned stemmed from an error in the file parsing stage of turning our dataset into a CSV, negligience on my end and should now be fixed. With this new fix, our MRE goes from 1.51% and 2.46% to 0.07% to 0.09% (primary and secondary respectively) giving us an avg improvement of 24.45%
- Also, I added a side by side comparison of KOI 1422's truth values vs predicted values, as well as their error. For radius, note that there is a difference in units which I'm lwk not gonna fix/convert until later.

### TO DO NEXT

- I'll be uploading the ANN's implementation on masking next, the code is done but I wanted two seperate git commits for such
- I also have to updated the way we return our Error evaluations, right now we return both linear and log radius, I'm just abiding by log for now, I'll fix this sometime eventually

### RESULTS

| Parameter | MRE |
|---|---|
| p_teff | 0.66% |
| s_teff | 0.70% |
| p_logg | 0.14% |
| s_logg | 0.14% |
| p_radius_log | 0.07% |
| s_radius_log | 0.09% |


| Parameter | Predicted | Truth | Error |
|---|---|---|---|
| p_teff | 3709.5198 | 3665 | 1.2% |
| s_teff | 3495.2395 | 3372 | 3.6% |
| p_logg | 4.8933 | 5.03 | -0.14 dex |
| s_logg | 4.9380 | 4.53 | +0.41 dex |
| p_radius_log | 24672921600 | 0.41 | 13.4% |
| s_radius_log | 28766507008 | 0.67 | 38.3 % |




![Predicted vs True Log g](/ANN-for-Stellar-Label-Determination/ANN/results/pred_vs_true_p_logg.png)

See the other files in folder ANN/results