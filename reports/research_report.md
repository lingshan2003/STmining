# Cross-City Few-Shot Traffic Forecasting with Spatiotemporal Graph Models

## Abstract

This project studies whether a forecasting model pretrained on a data-rich road-sensor graph can improve predictions on a different graph with only one, three, or seven days of observations. The repository provides a controlled comparison of non-spatial and graph-based models under scratch, zero-shot, fine-tuning, and temporal-freezing protocols.

## Research questions

1. Does explicit road-sensor connectivity improve single-city forecasting over node-shared temporal models?
2. Does source-city pretraining improve data efficiency in a target city?
3. Are temporal features more transferable than a fully fine-tuned spatiotemporal representation?

## Method

All models consume 12 five-minute observations and predict the next 12. The node-shared LSTM represents temporal-only transfer. STGCN alternates gated temporal convolution and graph convolution using a city-specific normalized adjacency matrix. Its learned weights operate on channels rather than node identities, enabling checkpoint transfer between different graph sizes.

## Experimental protocol

Use chronological 70/10/20 splits, fit normalization only on the training period, and evaluate 15-, 30-, and 60-minute horizons with masked MAE, RMSE, and MAPE. Run each few-shot experiment with seeds 42, 43, and 44. Generated tables belong in `reports/generated/`.

## Results and conclusion

Do not fill this section before running the experiments. Report mean ± standard deviation and explicitly discuss negative transfer rather than selecting only favorable runs.

