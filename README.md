### Updates explained:

- changed our error results from MAE and MRE to MSE, I was wrong :(
- okay basically i realized after talking with my super awesome FEMALE machine learning professor (who doesn't mansplain all the time) that I think we were going down an unneccessary rabit hole with analyzing our data, and instead we should be focusing on what the heck our machine learning model is doing as we are training and testing the ANN (she's super awesome). some relevant points:
    - our dataset might honestly still be too small, and if our science is correct, we should increase the dataset


- more things for me to remember: we are doing a custom loss function (criterion) because we wanted to place more emphasis on our secondary values, especially when faint. i might retrack that because i KNOW we're overfitting



### RESULTS

MSE: 1838.5778

Mean Squared Error per parameter:
- p_teff    : 5250.8159
- s_teff    : 5780.6496
- p_logg    : 0.0002
- s_logg    : 0.0003
- p_radius  : 0.0004
- s_radius  : 0.0004



![KOI Secondary Teff Luminosity Ratio Predictions](/ANN/filtered%20results/KOI's%20Secondary%20Teff%20(Color-coded%20by%20Luminosity%20Ratio).png)

You can see "yellow" or stars with a larger luninosity ratio, are predicted more accurately along the line, but stars with a lower, nearly faint ratio, are stuck at around 6000 K - 6500 K, as the model is overpredicting. I would also like to say that sometimes the accuracy of the yellow band varies based on our ANN, occassionally the model will overpredict but relatively stays pretty accurate.

![Train vs Validation Loss](/ANN/filtered%20results/Train%20vs%20Validation%20Loss.png)

Overall, pretty good for our Train vs Validation Loss, we're working with a smaller dataset so the spikey data makes a lot of sense here.