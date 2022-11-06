We have pickled the random forest model, which is our final model of choice. 

```
# To run preprocessing (edit the path to json data as required): 
python preprocessing.py ../data/data.json

# To run the modelling:
python random_forest.py -i preprocessed_data.csv
```
