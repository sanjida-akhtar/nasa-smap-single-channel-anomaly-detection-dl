# nasa-smap-single-channel-anomaly-detection-dl
The project employs LSTMs using Keras/Tensorflow to identify anomalies in multivariate sensor data for a NASA SMAP single channel, P-1. LSTMs are trained to learn normal system behaviors using encoded command information and prior telemetry values. Predictions are generated at each time step and the errors in predictions represent deviations from expected behavior. A nonparametric, unsupervised approach is used for thresholding these errors and identifying anomalous sequences of errors.
It's a small-scale re-implementation of the original paper,"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (https://arxiv.org/abs/1802.04431), which describes the background, methodologies, and experiments in more detail.
## Data
Data is collected from https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl. Data of channel_id P-1 is only used in this project. 
- Multivariate time-series data
- Preprocessed
## Methodology
This project follows the method described in the paper. Long Short-Term Memory (LSTM) is used to capture temporal patterns in the data.
### Model Overview
- Input: Sliding windows of time-series data, number of features
- Architecture:
    -LSTM layers
    -Dense layer (output layer)
- Loss function: Mean Squared Error 
Anomaliles are detected by computing prediction errors, smoothing errors and then applying a threshold.

## Results
The model is trained succesfully. However, obtained normalized mean absolute error value close to the value mentioned in the original paper. The model achieves a recall of 100%, a precision of 42.83% (~43%), and a F0.5 score of 48.39% without pruning false positives (close to the respective value mentioned in the paper). Currently attempts are being made to improve its performance.
## Limitations
Capturing continuous anomaly sequences is a challenging task. False positives are not pruned.
## References
- This project is inspired by Hundman et. al, "Detecting "Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", 2018
- Original work reference:
    - https://github.com/khundman/telemanom
## How to Run
### Prerequisite
- Python
- pip
- Jupyter notebook
### Steps
- Clone the repository
  ```bash
  git clone https://github.com/sanjida-akhtar/nasa-smap-single-channel-anomaly-detection-dl.git
- Install dependencies
  ```bash
  pip install -r requirements.txt
- Run jupyter notebook
