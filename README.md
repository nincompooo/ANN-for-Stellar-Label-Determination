### Updates explained:

- COMPLETELY REDID the ANN, now incorporates a feature in which automatically stops at the best # of epochs, mostly irrelevant and usually optimaizes at around 20 but it was mostly for me.
- Redid the appraoch to our ANN
	- Each spectrum is normalized by its own median flux, matching how the synthetic data was constructed and removes absolute scale so the network focuses on spectral shape.
    - THE ANN (4 hidden layers, w/ the output layer having 6 neurons (one per label).)
        - Hidden layers: progressively compress and reshape information, extracting nonlinear spectral features.
        - ReLU: adds non‑linearity (so the model can learn complex relationships).
        - BatchNorm: stabilizes training by normalizing activations layer‑wise.
        - Dropout: prevents overfitting by randomly dropping neurons.
    - training loop to minimize mean squared error
    - 

### TO DO NEXT

- honestly, our raidus is a bit wonky but that's very clearly because it's not consistent. I still have to go back and fix that, but that's a later problem.

### RESULTS

Median Relative Errors (linear radius):
p_teff    : 0.72%
s_teff    : 0.75%
p_logg    : 0.14%
s_logg    : 0.20%
p_radius_log: 1.51%
s_radius_log: 2.46%
Log p_radius (model output): 24.272856
Log s_radius (model output): 0.3447865

Predicted stellar parameters for new spectrum:
p_teff    : 4156.6772
s_teff    : 3237.5415
p_logg    : 4.7721
s_logg    : 5.0611
p_radius  : 34799046656.0000
s_radius  : 1.4117


![Predicted vs True Log g](/ANN/results/ANN-for-Stellar-Label-Determination/ANN/results/pred_vs_true_p_logg.png)
See the other files in folder ANN/results