import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Leer el archivo  CSV

data = pd.read_csv("sentiment140.csv", encoding="ISO-8859-1", usecols=[0, 5], names=["sentiment", "text"])

# 2. Colocamos en valor 1 si la sentimiento es positivo y 0 si es negativo

data['sentiment'] = data['sentiment'].replace(4, 1)  # Positivo = 1
data['sentiment'] = data['sentiment'].replace(0, 0)  # Negativo = 0

# 3. "X" contiene los textos que se van a clasificar. "Y" contiene las etiquetas de sentimiento asociadas
X = data['text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorización TF-IDF es una técnica para convertir texto en vectores numéricos teniendo en cuenta qué tan importante es una palabra en relación con el resto.
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Entrenar el modelo de regresion en este caso positivo/negativo en este caso
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Utiliza el modelo entrenado para predecir los sentimientos del conjunto de prueba
y_pred = model.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# 7. Guardar el modelo y el vectorizador
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
