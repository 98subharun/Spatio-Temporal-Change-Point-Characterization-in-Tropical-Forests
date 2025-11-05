Note:-This ReadMe only provides snippets of my actual latest paper. DOI to paper will be provided when the paper is published OR is on TechArxhive

# Description
The Amazons Forest sequesters vast amounts of carbon, making it one of the planet’s largest terrestrial carbon sinks, unfortunately, illegal deforestation and climate-related deterioration threaten this critical ecosystem. Current historical analysis of forests through satellite imagery often results in misclassification due to seasonal variations, cloud cover, and sensor anomalies. This study proposes a Novel hybrid pipeline for break point characterization using a Break for Additive Seasons and Trends (BFAST) time series algorithm combined with a Convolutional Neural Network (CNN) and Extreme Gradient Boosting (XGB) architecture trained on a sophisticated synthetic dataset leveraging Deep Learning techniques. This system gains an accuracy of ~97% in characterizing changes into 5 major categories demonstrating the impact of such a system in forest monitoring.

# Data acquisition and preprocessing
Spectral Trajectories were simulated for individual pixels, the baseline was constructed using sinusoidal seasonal curves with realistic tropical forest periodicity and random noise was interjected to replicate sensor artifacts, cloud contamination, and natural variability. The classes were modelled as, Major Deforestation which was represented by abrupt changes/ Permanent decline in NDVI/NBR, Moderate deforestation which was represented by partial Canopy loss, Minor degradation was also a chosen class. Regrowth/Recovery showed a Gradual increase following a disturbance, Stable Forest which was represented by consistent seasonal cycles which were perpetuated only by noise and random outliers such as any cloud shadowing, atmospheric interference, etc. These outliers are critical for the testing of the formulation of the Huber regression and Savitzky-Golay Smoothing.

Huber regression is given by:-

It is quadratic for small residuals and linear for large residuals:

$$
L_\delta(y, f(x)) =
\begin{cases}
\frac{1}{2}(y - f(x))^2, & \text{if } |y - f(x)| \le \delta \\
\delta \left( |y - f(x)| - \frac{1}{2}\delta \right), & \text{otherwise}
\end{cases}
$$

Where:  
- $y$: true value  
- $f(x)$: model prediction  
- $\delta$: threshold controlling the transition between quadratic and linear regions  

The regression minimizes:

<div align="center">
  <img src="https://latex.codecogs.com/png.latex?\dpi{150}\color{white}\LARGE\min_{w,b}\sum_{i=1}^{n}L_\delta(y_i,w^Tx_i+b)" alt="Huber Regression Formula"/>
</div>




Huber regression is robust to outliers and more stable than L1 regression.

---

Now, Savitzky-Golay here smooths the data while preserving peak shape.  
It fits a local polynomial of degree $p$ within a moving window of size $2m + 1$.

At each central point $x_0$:

$$
y(x) \approx a_0 + a_1(x - x_0) + a_2(x - x_0)^2 + \dots + a_p(x - x_0)^p
$$

and the smoothed estimate is:

$$
\hat{y}(x_0) = a_0
$$

The coefficients are computed via least squares:

$$
\mathbf{a} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

where $\mathbf{X}$ is the Vandermonde matrix of local points.

---

# BFAST
A classical Breaks for Additive Seasonal and Trend (BFAST) was modified to address challenges specific to tropical forests. A Huber Regression system was adopted to reduce sensitivity to outliers from cloud cover and sensor noise. Savitzky-Golay smoothing was adopted to smooth the residuals while still preserving the breakpoints. Density based outliers were with distinct multipliers for dense, medium, and sparse forests to improve the classification sensitivity. Change points were only confirmed if they were detected across a variety of indices, including NDVI, NBR and NDMI as each index has different ecological parameters, so combining them would lead to lower misclassification and capture a broader range of real changes.
This design allows the model to capture not just the presence of some form of change but also its persistence, producing descriptors, recovery trajectories and the severity of canopy distribution. This allows the BFAST to act as a temporal backbone on the system.
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large Y_t=T_t+S_t+e_t" />
</div>

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large T_t=\alpha_0+\alpha_1t+\sum_{k=1}^{K}\beta_kD_k(t)" />
</div>

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large S_t=\sum_{j=1}^{J}\left(\gamma_{1j}\sin\frac{2\pi jt}{f}+\gamma_{2j}\cos\frac{2\pi jt}{f}\right)" />
</div>
# Spatial and spectral CNN embeddings
The CNN was trained on synthetic dataset generated from the BFAST with each sample encoding complex vegetation dynamic derived from the simulated NDVI, NBR, NDMI and EVI signals. This setup provided precise supervision across all the change classes. The advantage of this is that it provides precise supervision. The CNN embeddings learn features which are tied to well characterized disturbance patterns as the synthetic generator defines both the timing and type of change. The CNN was implemented as a series of sequential convolutional and pooling layers, moreover the architecture was made to be lightweight which prevents overfitting. The final layer produced a dense feature vector which was approximately 512 dimensions in this prototype which encoded abstract attributes. These were later concatenated with the temporal descriptors derived from the BFAST and ancillary spatial features. Seasonal cycles, stochastic noise and random disturbances were injected into the simulated dataset itself thus forcing the CNN to generalize across a wide range of unfavorable and favorable conditions.


<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large h_l=f(W_l*h_{l-1}+b_l)" />
</div>

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large z=\text{GlobalAvgPool}(h_L)" />
</div>



# XGB
The feature vectors from the BFAST and CNN were concatenated and later classified by the XGB. The input feature vector consisted of 20+ time-based descriptors taken from the BFAST pipeline which includes seasonal amplitudes, recovery duration, inter index correlation and break persistence. The synthetic dataset was generated using canopy disturbances into a Landsat like time series. The model’s hyper parameters were later tuned to balance generalization and sensitivity. Bias-variance tradeoff was controlled through learning rate tuning and stochastic subsampling so that the model generalized effectively, randomness was introduced to reduce the correlations between the individual trees in the model, addressed imbalances between change and no change A cross-index voting system was enforced to consider a joint index response. For example: - A true response was only considered if the NDVI decline was consistent with NBR loss and NDMI moisture shifts. This reduced false positives by ~25%.
The final output was a pixel wise probability map of the forests change, with confidence intervals derived from variance. By providing pixel wise probability maps with confidence intervals, the system could not only identified disturbance but also quantify uncertainty.


<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large \hat{y}_i=\sum_{k=1}^{K}f_k(x_i),\quad f_k\in\mathcal{F}" />
</div>

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large \mathcal{L}^{(t)}=\sum_i l(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)" />
</div>

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\color{white}\Large \Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j}w_j^2" />
</div>



(Results and discussion section will be provided later)
