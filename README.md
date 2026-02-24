### Updates explained:

- I finally understand where I went wrong with radii, and it was a completely silly mistake. 

### TO DO NEXT

- optimization, rn log g seconary returns higher outputs than secondary, maybe fiddle with the ANN to put restraints on that.

### RESULTS

| Parameter | MRE |
|---|---|
| p_teff | 0.67 |
| p_logg | 1.05% |
| s_logg | 0.15% |
| p_radius | 1.31% |
| s_radius | 3.42% |


| Parameter | Predicted | Truth | Error |
|---|---|---|---|
| p_teff | 3931.0134 | 3665 | +7.26% |
| s_teff | 3353.1851 | 3372 | −0.56% |
| p_logg | 4.8301 | 5.03 | −3.98% |
| s_logg | 5.0078 | 4.53 | +10.55% |
| p_radius | 0.4556 | 0.41 | +11.12% |
| s_radius | 0.2739 | 0.274 | −0.04% |




![Predicted vs True Log g](/ANN-for-Stellar-Label-Determination/ANN/results/pred_vs_true_p_logg.png)

See the other files in folder ANN/results