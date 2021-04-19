# Esercizi pratici

## Passaggi preliminari
Entra nella directory del repository e crea un nuovo virtual env. 
```sh
$python3 -m venv ./venv_corso
```
Installa i requirements del repository e i pacchetti locali.
```sh
$pip install -r requirements.txt
$pip install -e .
```


## Creazione di un test
Il file *tests/test_data_and_model.py* contiene un esempio di test scritto con pytest. Prova a scrivere un altro test che importi il modello serializzato e:
* Controlli che il classificatore non sia un 'majority classfifier', ovvero che sia in grado di classificare piu' di un etichetta sul test set
* Controlli che la precisione e la sensitivita' (recall) del modello siano sopra una certa soglia da te scelta.
<details> 
  <summary>Possibile soluzione</summary>

    def test_model_metrics(adult_test_dataset):
        x, y, data_path = adult_test_dataset
        clf = joblib.load('./model.pkl')
        predictions = clf.predict(x)
        metrics = classification_report(y, predictions, output_dict=True)
    
        assert len(np.unique(predictions)) > 1
        assert metrics['>50K']['precision'] > 0.7 #fill here
        assert metrics['>50K']['recall'] > 0.1 #fill here
</details>

## Creazione di una GitHub Action
Crea una cartella chiamata '.github' all'interno della directory principale. All'interno di questa cartella crea una cartella chiamata 'workflow'.

In quest'ultima crea un file 'CI.yaml' e copia/incolla il seguente codice
```yaml

name: Test

on: [push]

jobs:
    build:
        runs-on: ubuntu-latest
        strategy:
            matrix:
                python-version: [3.7, 3.8]

        steps:
        - uses: actions/checkout@v2

        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v2
          with:
            python-version: ${{ matrix.python-version }}

        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            pip install -e .

        - name: Pytest
          run: |
            pytest -v --maxfail=3 --cache-clear
```
Effettua un commit e un push e segui la action direttamente su GitHub (repository --> tab 'actions')

## Ricerca iperparametri con mlflow
Modifica lo script ./experiments/run_grid_search.py cambiando lo spazio di ricerca (aggiungendo iperparametri).
Una volta arricchito la spazio di ricerca fai girare lo script  
```sh
$python experiments/run_grid_search
```
Quali sono



## Esercizio 4
