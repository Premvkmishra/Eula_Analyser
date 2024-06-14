from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os

eulas = ["EULA text 1", "EULA text 2", ...]  
labels = [0, 1, ...]  # 0: benign, 1: harmful

from preprocessing import preprocess_text
eulas = [preprocess_text(eula) for eula in eulas]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(eulas)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
with open('models/svm_model.pkl', 'wb') as file:
    pickle.dump((vectorizer, model), file)
