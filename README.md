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
Il file tests/test_data_and_model.py contiene un esempio di test scritto con pytest.
<details> 
  <summary>Possibile soluzione</summary>

df
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
Modifica lo script ./experiments/run_grid_search.py cambiando lo spazio di ricerca (aggiungendo iperparametri)
Fai girare lo script  
```sh
$python experiments/run_grid_search --name python
```




## Esercizio 4
