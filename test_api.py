import pytest
from flask import Flask
import pandas as pd
import random
from api import app 

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture
def random_test_data():
    # Lire le fichier CSV
    df = pd.read_csv('X_test50_woIndex.csv')
    # Sélectionner une ligne au hasard
    random_row = df.sample(n=1).to_dict(orient='records')[0]
    return random_row

def test_prediction_real_model(client, random_test_data):
    # Envoyer une requête POST à la route /prediction avec les données de test aléatoires
    response = client.post('/prediction', json=random_test_data)

    # Vérifier que la réponse a le statut 200
    assert response.status_code == 200

    # Vérifier le contenu de la réponse
    json_data = response.get_json()
    assert 'prediction' in json_data
    # Effectuer des vérifications sur json_data en fonction des prédictions attendues