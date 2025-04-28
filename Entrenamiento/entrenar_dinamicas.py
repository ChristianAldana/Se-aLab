import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Cargar los datos
csv_path = 'utils/data/landmarks.csv'
df = pd.read_csv(csv_path)

# Separar características (X) y etiquetas (y)
X = df.drop('label', axis=1)
y = df['label']

# Separar datos para entrenar y probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluar el modelo
print(f"Precisión: {model.score(X_test, y_test) * 100:.2f}%")

# Guardar el modelo
modelo_path = 'Letras/modelo_abecedario.pkl'

# Crear la carpeta Letras si no existe (por seguridad)
os.makedirs('Letras', exist_ok=True)

joblib.dump(model, modelo_path)

print(f"Modelo guardado en {modelo_path} ✅")
