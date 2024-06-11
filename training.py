from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os

# Example data and labels (replace with actual data)
eulas = ["EULA text 1", "EULA text 2", ...]  # Load your EULA texts
labels = [0, 1, ...]  # 0: benign, 1: harmful

# Preprocess the texts
from preprocessing import preprocess_text
eulas = [preprocess_text(eula) for eula in eulas]

# Vectorize the texts
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(eulas)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the model
model = SVC()
model.fit(X_train, y_train)

# Save the model and vectorizer
os.makedirs('models', exist_ok=True)
with open('models/svm_model.pkl', 'wb') as file:
    pickle.dump((vectorizer, model), file)
