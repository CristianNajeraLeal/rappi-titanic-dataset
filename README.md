# Rappi - Titanic Dataset

---

### How to use the package

First step `pip install -r requirements.txt`

The package is designed to be run with a single command `make all` and it will run the training pipeline and locally
deploy the model. It will create an MLFlow experiment named `Titanic` for the purpose of this run.

In case of wanting to change the training execution you have four options:
1. --experiment-name
   1. To change the experiment name
2. --run-name
   1. To change the run name
3. --data-path
   1. To change the data path
4. --log-plot
   1. To save or not the confusion matrix plot. Default=True

Example:
`python train.py --experiment-name experiment2 --run-name run2 --data-path new_data_path --log-plot False`

If you changed the experiment name on the training, you will need to change it in the run too.

In the run scenario you have three options:
1. --image-name
   1. To change the Docker image to be generated. Must be all lowercase.
2. --experiment-name
   1. To change the experiment name
3. --port
   1. To change the endpoint port

Example:
`python run.py --image-name image2 --experiment-name experiment2 --port 1234`


---

### To test the endpoint

1. Go to Postman
2. Endpoint `http://localhost:5001/invocations`
3. Method POST
4. Body (raw)
```json
{
    "dataframe_records": [
        {
            "PassengerId":891,
            "Pclass":1,
            "Name":"Kelly, Mr. James",
            "Sex":"male",
            "Age":15,
            "SibSp":1,
            "Parch":0,
            "Ticket":"330911",
            "Fare":57.8292,
            "Cabin":null,
            "Embarked":"S"
        }
    ]
}
```

Another option is a `curl` call from the terminal
```bash
curl -X POST http://localhost:5001/invocations \
-H "Content-Type: application/json" \
-d '{
    "dataframe_records": [
        {
            "PassengerId": 891,
            "Pclass": 1,
            "Name": "Kelly, Mr. James",
            "Sex": "male",
            "Age": 15,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "330911",
            "Fare": 57.8292,
            "Cabin": null,
            "Embarked": "S"
        }
    ]
}'
```

It will return 
```json
{"predictions": [0]}
```

Note: If you changed the endpoint port, please replace it on the curl or Postman.

---

### Coverage Report

You can find the coverage report `index.html` on the `htmlcov` folder.
Also, you can run `make test` to generate it.


---

### Notebook

On the `notebooks` folder, it is called `titanic_dataset.ipynb`. You may need to install Jupyter to check this.