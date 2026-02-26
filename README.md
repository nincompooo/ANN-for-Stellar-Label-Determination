### Updates explained:

- Updated SNR and filtered out 83 stars with Teff's above 6200 K leaving us with 195 stars.

### RESULTS

Median Relative Errors per parameter:
p_teff    : 1.32%
s_teff    : 2.65%
p_logg    : 0.23%
s_logg    : 0.60%
p_radius  : 2.62%
s_radius  : 8.24%

As you can see with KOI 1422, it aligns quite well with what we can be assured are valid estimated values, from both Kendall's model and my ANN

KOI 1522 Truth stellar parameters:
p_teff    : 3665
s_teff    : 3372
p_logg    : 5.03
s_logg    : 4.53
p_radius  : 0.41
s_radius  : 0.274

KOI 1422 Predicted stellar parameters:
p_teff    : 3771.2600
s_teff    : 3333.8699
p_logg    : 4.8240
s_logg    : 5.0250
p_radius  : 0.4568
s_radius  : 0.2585


![Predicted vs True Primary Teff](/ANN/results/pred_vs_true_p_teff.png)
![ANN vs Derived Primary Teff](/ANN/Prediction_Plots/pred_vs_true_p_teff.png)
![ANN vs Derived Secondary Teff](/ANN/Prediction_Plots/pred_vs_true_s_teff.png)