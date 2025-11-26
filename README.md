This code is a re-written version of [fit-rec](https://github.com/nijianmo/fit-rec) in pytorch so far.

# Running Experiment
1. Run `cd data; wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json` to download the necessary data
2. Run 'python3 data_split.py'
3.
  ```python3 heart_rate_aux.py \
  --epoch 50 \
  --batch_size 256 \
  --attributes userId,sport,gender \
  --input_attributes distance,altitude,time_elapsed \
  --temporal
  ```

> When running failed before data processing, `/data/processed_endomondoHR_proper.npy` may needs to be removed before restarting.
