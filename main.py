from flask import Flask, request, render_template
from analyzer import analyze_eula

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    eula_text = request.form['eula_text']
    prediction = analyze_eula(eula_text)
    result = "Harmful" if prediction == 1 else "Benign"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
