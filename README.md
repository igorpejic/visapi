# Monte Carlo Tree Search on Perfect Rectangle Packing Problem Instances

![GitHub Logo](/images/perfect_rectangle_packing.png)

## Dataset
The dataset of 1000 Guillotinable and 1000 Braam rectangle packing problems with 20 tiles can be found in [problems](problems/).

## Installation
```
pip3 install requirements.txt
```

To run experiments, you will need to have a PostgreSQL database running.

Copy the file `visapi/secrets.json.local` to `visapi/secrets.json`:

```
cp visapi/secrets.json.template visaip/secrets.json
```

Then edit it with a secret key and the details to connect to the database:
```
The database values meanings are in order:
HOST, DATABASE_NAME, USER, PASSWORD, PORT
```

You will need to have such a user with that password and a database with DATABASE_NAME created before being able to run experiments.
All experiments results will be saved to the database.

## Running experiments

### MCTS algorithm
From file:
```
PYTHONPATH=engine/ python3 manage.py run_mcts 20 11 11 --from_file
```

Dynamically generated problems:
```
PYTHONPATH=engine/ python3 manage.py run_mcts 20 11 11
```


### Neural Network algorithm

This algorithm performs only one tile placement.
To get the final solution calls the NN predict multiple times.

To train the neural network you can use:
```
PYTHONPATH=engine/ python3 manage.py just_train                                                  
```
