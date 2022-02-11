from flask import Flask

from textblob import TextBlob

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/1576-mlops-machine-learning/aula-5/casas.csv')
colunas = ['tamanho', 'preco']
df = df[colunas]

x = df.drop('preco', axis=1)
y = df['preco']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
model = LinearRegression()

model.fit(train_x, train_y)


app = Flask(__name__)

@app.route('/')
def home():
  return "Minha segunda API."

@app.route('/sentimento/<frase>')
def sentimento(frase):
  tb = TextBlob(frase)
  tb_en = tb.translate(to='en')
  polarity = tb_en.sentiment.polarity
  
  return f"Polaridade: {polarity}"

@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
  preco = model.predict([[tamanho]])[0]
  return f'Preço: {preco}'


app.run(debug=True)