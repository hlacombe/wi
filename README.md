# Click prediction

## launch the project
The project provides 2 commands:
* **train:** train the models on dataset.json
* **predict:** predict labels on the dataset and create a csv in the output folder thanks to the models

All models are stored in the *model* folder. We provides this folder with pre-trained models.

```shell script
sbt
run predict <path to json dataset>
```
