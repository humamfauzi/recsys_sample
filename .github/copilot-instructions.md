# structure
There are three main folder. First is train folder which contains all training sequence from loading data, training data, and saving model.
Second is the serve which serving model. Contains all webserver and ability to load a model.
Third is intermediaries. This folder accessible from train and serve but not require from train and serve.
the data is in dataset folder

# train
We use scenario to control the the training flow. The main executable for the train module is the `main.py` we pass this following flag to control the training flow. We dont have much variation here. So we use the default flag.

All data reading and saving are done via `io.py` which reads the data and turn it to a numpy array. It also able to save the weight to a file which can be read later by serve module.
In this recommendation system, reading data would yield three different array. users array which contains user id and metadata, product array which contains product id and product metadata
and the rating table which connect users, product, and the rating. `io.py` contain two class. One class contain the actual implementation and one class is an interface that tells other class what methods available for other class.
These classes should be tested against a dummy data; ensuring that if it can read 10 row, then it would be able to read 10000 rows.

After data are read and transformed into a numpy, a `validator.py` take those numpy array and validate it. There are three things we need to validate
1. validate that every user in rating table exists in user table
2. validate that every product in rating table exists in product table
3. Remove all null rating value.
If we have more ideas about data validation, we would put it here. All validation should be tested.

Once validation is done, we can sure that our data wont break our calcluation. We need to prepare the data in `pre.py` before feeding it to training model. There are few things that need to be done.
1. encoding the metadata to be its own column so we can assign the weight
2. merging all three different array into one large array
3. separate the data to training and test. The ration decided in input. It should be randomized.
4. separate the training data into fold. The number of fold decided in input. It should be randomized. the result of the fold should be an index
So the final result should an array of training data set, test data set, and fold index. This result should be a data class.

Once all data is set, we can train it in `train.py`. This train have input of the data class created by `pre.py`. For current iteration, the main class in `train.py` would train the rating using ALS. 
the ALS itself should be on separate class. It should took the training data and paramters. We can set the list of possible parameters like latent factor number, regularization in the main class.
the ALS class have resposible to initiate each weight and user and product bias. in main class we can call fit to find the best parameters. Parameters, weight, map user and product to its respective index and global mean should returned as a single data class.

`io.py` can receive this data class and store it as a JSON (if you have better format then use it). Store it in folder called models

# serve
like in train, main executable is `main.py`. In this file, we start the webserver using FastAPI. We read the models first. Serve have their own `io.py` file to read the models. The reader is under a class as well.
It also has `model.py` which contain a class that has input of the model loaded from `io.py`. This class can read model and reconstruct the model in such a way that if we feed the user and its metadata, how many recommendation we want to generate, it would return a list of recommended movies.
In `main.py` the class model from `model.py` act as the main model. An endpoint would use its inference to return recommended film.

all file, `model.py` and `io.py` should have test unit.

# intermediaries
this contain the definition, enums, and class that can be used by both train and serve. If there is a similar class that required in both, then move it to intermediaries and let serve and train require from intermediaries.

# other files
- We add gitignore to not commit dataset and models
- We add makefile so we can save command to train and serve. It also contain command to compile both train and serve into a binary
- All makefile command should be included in PHONY

# test
In testing use `unittest` framework. All test should be in `tests/` folder. The test should be able to run independently.

# reference
https://cookiecutter-data-science.drivendata.org/
https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
https://cs231n.stanford.edu/slides/2016/winter1516_lecture5.pdf
https://karpathy.github.io/2019/04/25/recipe/
https://zeabur.com/servers