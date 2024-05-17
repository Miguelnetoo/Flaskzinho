from flask import Flask, request, render_template
import pandas as pd
import pickle

# Carregar o modelo de Regressão Linear e a base de dados
modelo = pickle.load(open('modelo_regressao_linear.pkl', 'rb'))
dados = pd.read_csv('Salary_Data.csv')

app = Flask(__name__)

@app.route('/') 
def index():
    return render_template('forms_salario.html')

@app.route('/predict', methods=['POST']) 
def predict():
    # Obter a experiência em anos do formulário
    experiencia = float(request.form['YearsExperience'])
    
    # Fazer a predição
    salario_predito = modelo.predict([[experiencia]])[0]
    
    return f'O salário previsto para {experiencia} anos de experiência é de: R$ {salario_predito:.2f}'

if __name__ == '__main__':
    app.run(debug=True, port=5000)