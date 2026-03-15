import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# 1. Carregar dados
df = pd.read_csv('/home/tostesx/codeJS/Inteligência Artificial/Atividade02/dataset_alunos_evasao.csv')

# 2. Separar features (X) e target (y)
X = df.drop('status_curso', axis=1)
y = df['status_curso']

# 3. Codificar variáveis categóricas
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 4. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Treinar modelo Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Avaliar modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Salvar modelo, encoders e nomes das features
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/modelo.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("Modelo treinado e salvo com sucesso!")