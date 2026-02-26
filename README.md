### Updates explained:

- See in ANN/cKOI_Analysis.py how I parse through which files in our KOI Spec 1, 2, and 3 files are in our ANN/Derived Star.tex with affiliated predicted Teff and Radius values.

### RESULTS

Not good. These are the metrics we trained our model on, and used to predict all the KOI values.

| Parameter | MRE |
|---|---|
| p_teff | 0.93% |
| s_teff | 1.30% |
| p_logg | 0.15% |
| s_logg | 0.27% |
| p_radius | 1.62% |
| s_radius | 3.83% |

As you can see with KOI 1422, it aligns quite well with what we can be assured are valid estimated values, from both Kendall's model and my ANN

| Parameter | Predicted | Truth |
|---|---|---|
| p_teff | 3819.8118 | 3665 |
| s_teff | 3359.6536 | 3372 |
| p_logg | 4.8371 | 5.03 |
| s_logg | 5.0052 | 4.53 |
| p_radius | 0.4530 | 0.41 | 
| s_radius | 0.2692 | 0.274 |


![Predicted vs True Primary Teff](/ANN/results/pred_vs_true_p_teff.png)

See the other files in folder ANN/results

HOWEVER, when we look at our prediction comparison results for the rest, results vary drastically, and worse for our secondary Teff values. I wanted to make sure it wasn't an issue with my plot function and double checked with all the tables in our Predicted Tables folder, and there are genuine LARGE inconsistencies between what our ANN predcited versus our previous model.

![ANN vs Derived Primary Teff](/ANN/Prediction_Plots/pred_vs_true_p_teff.png)
![ANN vs Derived Secondary Teff](/ANN/Prediction_Plots/pred_vs_true_s_teff.png)