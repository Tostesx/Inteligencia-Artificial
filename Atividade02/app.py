from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carregar modelo e artefatos
model = joblib.load('models/modelo.pkl')
encoders = joblib.load('models/encoders.pkl')
feature_names = joblib.load('models/feature_names.pkl')

@app.route('/')
def home():
    # Exibe formulário com campos para cada feature
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Coletar valores do formulário
    dados = []
    for feat in feature_names:
        valor = request.form[feat]
        # Se for categórica, codificar
        if feat in encoders:
            # Tenta transformar; se falhar, assume -1
            try:
                valor_cod = encoders[feat].transform([valor])[0]
            except:
                valor_cod = -1
            dados.append(valor_cod)
        else:
            # Assume numérico
            dados.append(float(valor))
    
    # Criar DataFrame com os nomes das colunas
    entrada = pd.DataFrame([dados], columns=feature_names)
    
    # Prever
    pred = model.predict(entrada)[0]
    prob = model.predict_proba(entrada)[0]
    
    resultado = "Evasão" if pred == 1 else "Conclusão"
    prob_evasao = prob[1]
    
    return render_template('result.html', 
                           resultado=resultado,
                           probabilidade=prob_evasao,
                           dados=request.form)

if __name__ == '__main__':
    app.run(debug=True)