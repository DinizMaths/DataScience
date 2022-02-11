from flask import Flask, request, jsonify

from textblob import TextBlob

from sklearn.linear_model import LinearRegression

import pickle


model = pickle.load(open('../model.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)

@app.route('/')
def home():
  return "Minha quarta API."

@app.route('/sentimento/<frase>')
def sentimento(frase):
  tb = TextBlob(frase)
  tb_en = tb.translate(to='en')
  polarity = tb_en.sentiment.polarity
  
  return f"Polaridade: {polarity}"

@app.route('/cotacao/', methods=['POST'])
def cotacao():
  dados = request.get_json()
  dados_input = [dados[col] for col in colunas]
  preco = model.predict([dados_input])

  return jsonify(preco=preco[0])


app.run(debug=True)