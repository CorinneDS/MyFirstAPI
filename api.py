from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

# Route pour afficher la dernière ligne reçue dans le navigateur
@app.route('/test', methods=['POST'])
def index():

    last_row_received = request.get_json()

    print("TYPE: ",type(last_row_received))

    # Construire une chaîne de caractères représentant la dernière ligne
    last_row_str = ' // '.join([f"{key}: {value}" for key, value in last_row_received.items()])

    print(last_row_str)

    # Afficher la dernière ligne dans le navigateur
    return jsonify(f"<h1>Derniere ligne recue :</h1><p>{last_row_str}</p>")

# Route pour afficher le resultat de la prediction
@app.route('/prediction', methods=['POST'])
def prediction():
    # Récupérer la ligne envoyée par le client depuis la requête JSON
    row_data = request.get_json()

    # Charger le modèle depuis le fichier Joblib
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model_best_LGBM.sav')
    best_model = load(model_path)

    # Vérifier si row_data est un dictionnaire
    if isinstance(row_data, dict):
        # Convertir le dictionnaire en DataFrame Pandas
        df = pd.DataFrame([row_data])
    
        prediction = best_model.predict_proba(df)
        prediction_finale = best_model.predict(df)

        combined_predictions = []

        # Ajouter les résultats de prediction à la liste combinée
        combined_predictions.extend(prediction.tolist())

        # Ajouter les résultats de prediction_finale à la liste combinée
        combined_predictions.extend(prediction_finale.tolist())

        # Retourner le résultat de la prédiction
        return jsonify({'prediction': combined_predictions})
    else:
        # Retourner un message d'erreur si le format n'est pas correct
        return jsonify({'error': 'Le format des données envoyées est incorrect. Assurez-vous d\'envoyer un objet JSON contenant les données de la ligne à prédire.'}), 400

if __name__ == '__main__':
    
    #app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

