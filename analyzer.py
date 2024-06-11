import pickle
from preprocessing import preprocess_text

def analyze_eula(eula_text):
    with open('models/svm_model.pkl', 'rb') as file:
        vectorizer, model = pickle.load(file)
    processed_text = preprocess_text(eula_text)
    X = vectorizer.transform([processed_text])
    prediction = model.predict(X)
    return prediction[0]
