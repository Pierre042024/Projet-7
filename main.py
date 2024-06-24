from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

# Charger le modèle enregistré
model = joblib.load('best_model.joblib')

# Charger les données des clients avec SK_ID_CURR comme index
try:
    clients_df = pd.read_csv('sampled_dataset.csv')
except ValueError as e:
    raise ValueError(f"Error reading CSV file: {e}")

# Initialiser l'application FastAPI
app = FastAPI()

# Route pour vérifier l'état de l'API
@app.get("/")
def read_root():
    return {"status": "API is running"}

# Route pour afficher les scores de précision du modèle
@app.get("/model_scores")
def model_scores():
    # Placeholder scores
    scores = {
        "auc": 0.74,
        "accuracy": 0.93,
    }
    return scores

# Route pour obtenir la liste des IDs des clients
@app.get("/clients")
def get_client_ids():
    return clients_df['SK_ID_CURR'].tolist()

# Route pour faire une prédiction basée sur l'ID du client
@app.get("/predict/{client_id}")
def predict(client_id: int):
    # Vérifier si le client existe
    if client_id not in clients_df['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Extraire les données du client
    client_data = clients_df[clients_df['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])
    
    # Faire une prédiction
    prediction = model.predict(client_data.values.reshape(1, -1))
    prediction_proba = model.predict_proba(client_data.values.reshape(1, -1))[:, 1]

    # Retourner la prédiction
    return {
        "prediction": int(prediction[0]),
        "probability": float(prediction_proba[0])
    }

# Exécution du serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)