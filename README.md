### Updates explained:

- Here we implement masking, to see results without masking, refer to the previous git commit. Additionally, we reference the[LRLRPayne: Stellar parameters and abundances from low-resolution spectra](https://arxiv.org/pdf/2511.06546) with our masking approach in which we add a wavelength mask function after loading the wavelength in our 'ANN2.py' file. However we don't interpolate or extrapolate because:
    - mask keeps same bins
    - scaler expects masked shape
    - ANN input matches exactly
- This differes from the LRPayne since the mask during χ² determination whereas we mask "inside" the ANN (technically before the ANN forward pass, like we do with injection). You can see what the masked spectrum looks like in. 
- Also note to self, we write masking function in microns, and we keep to the EXACT values Kendall gave us and DON'T pad Angstroms on both sides. 

0.6859282598158023 0.6860227823557925
0.6879132331555987 0.6880077556955889
0.7599394086282114 0.7600339311682016
0.7659888511875909 0.7660833737275813
0.8209064469219584 0.8210009694619487
0.8239311682016481 0.8240256907416383

![Masking Check](/ANN-for-Stellar-Label-Determination/ANN/results/masking_check.png)


### TO DO NEXT

- I also have to updated the way we return our Error evaluations, right now we return both linear and log radius, I'm just abiding by log for now, I'll fix this sometime eventually

### RESULTS

| Parameter | MRE |
|---|---|
| p_teff | 0.85% |
| s_teff | 0.81% |
| p_logg | 0.13% |
| s_logg | 0.22% |
| p_radius_log | 0.05% |
| s_radius_log | 0.12% |


| Parameter | Predicted | Truth | Error |
|---|---|---|---|
| p_teff | 3892.3513 | 3665 | +6.20 % |
| s_teff | 3711.8232 | 3372 | +10.07 % |
| p_logg | 4.8843 | 5.03 | −0.146 dex |
| s_logg | 4.9140| 4.53 | +0.384 dex |
| p_radius_log | 25460154368 | 0.41 | ?? |
| s_radius_log | 24998350848 | 0.67 | ?? |




![Predicted vs True Log g](/ANN-for-Stellar-Label-Determination/ANN/results/pred_vs_true_p_logg.png)

See the other files in folder ANN/results