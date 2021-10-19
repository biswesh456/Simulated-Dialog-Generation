# Agent Model

Uses the dialogpt as an end to end model to train on multiwoz dataset. 

## Training

python train.py  --data 'DATA_DIR' --base-dir 'BASE_DIR' --model 'MODEL_NAME' --mname 'specific name'

DATA_DIR - place containing data relative to base directory
MODEL_NAME - provide name of model which would store models

Model and Data folders need to be created in the base directory

## Testing

python3 test.py  --data 'DATA_DIR' --base-dir 'BASE_DIR' --model 'MODEL_NAME' --mname 'specific name'

Predictions folder need to be created in the base directory in order to store the predictions.






