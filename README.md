# Monte Carlo Tree Search on Perfect Rectangle Packing Problem Instances

![GitHub Logo](/images/perfect_rectangle_packing.png)

## Dataset
The dataset of 1000 Guillotinable and 1000 Braam rectangle packing problems with 20 tiles can be found in [problems](problems/).

## Installation
```
pip3 install requirements.txt
```

To run experiments, you will need to have a PostgreSQL database running.

After installing the PostgreSQL database you can create one by logging into the psql shell and running:

```psql
postgres=# create role binpack_user;
CREATE ROLE
postgres=# create database binpack;
CREATE DATABASE
postgres=# grant all on database binpack to binpack_user;
GRANT
postgres=# alter role binpack_user with login;
postgres=# alter user binpack_user with password 'password';
```

Copy the file `visapi/secrets.json.local` to `visapi/secrets.json`:

```
cp visapi/secrets.json.template visaip/secrets.json
```

Then edit it with a secret key and the details to connect to the database:
```
The database values meanings are in order:
HOST, DATABASE_NAME, USER, PASSWORD, PORT
```

For the example provided, the contents of the file would be:
```json
{
    "SECRET_KEY": "something_random",
    "DATABASE": ["localhost", "binpack", "binpack_user", "password", ""]
}
```

To create a database schema, you will need to run:

```shell
python3 manage.py migrate
```

All experiments results will be saved to the database, in the table `affinity`.

Access them using the django shell:
```
python3 manage.py shell_plus
```

And then:

```python
Result.objects.all()
```

See Django ORM [reference](https://docs.djangoproject.com/en/3.1/topics/db/queries/#making-queries).


Or access them using postgresql shell:
```psql
select * from affinity;
```

All the problem results will be available in the database with all the metrics needed to finish them (number of tiles placed etc.), and their scores (success / failure).

## Running experiments

### MCTS algorithm
Run:

```
PYTHONPATH=engine/ python3 manage.py run_mcts
```

to see all the options for the runner.

Example: 

Run dynamically generated problems:
```
PYTHONPATH=engine/ python3 manage.py run_mcts 20 11 11
```

From file:
```
PYTHONPATH=engine/ python3 manage.py run_mcts 20 11 11 --from_file
```



### Neural Network algorithm

This algorithm performs only one tile placement.
To get the final solution calls the NN predict multiple times.

To train the neural network you can use:
```
PYTHONPATH=engine/ python3 manage.py just_train                                                  
```
