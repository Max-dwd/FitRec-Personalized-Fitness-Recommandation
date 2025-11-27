Run `cd data; wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json` to download the necessary data

Run 'python3 data_split.py'

Run `python3 heart_rate_aux.py   --epoch 50   --batch_size 512  --attr_dim 5   --hidden_dim 64   --attributes userId,sport   --input_attributes distance,altitude,time_elapsed   --temporal`