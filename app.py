from flask import Flask, render_template, request
import pickle

# Cargar el modelo y vectorizador
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocesar y predecir
        vectorized_text = tfidf_vectorizer.transform([text])
        prediction = model.predict(vectorized_text)
        
        # Mapear el resultado
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        
        return render_template('index.html', prediction_text=f'El sentimiento del comentario es: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
