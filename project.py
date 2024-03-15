from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

data = pd.read_csv('base1.csv').dropna()

X = data.drop(columns=["game_name", "game_image"])
y = data["game_name"]

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggested', methods=['POST'])
def suggested():

    age = int(request.form['age'])
    gender = int(request.form['gender'])

    input_data = [[age, gender]]
    WhatGame = model.predict(input_data)[0]

    game_img = f"images/{data[data['game_name'] == WhatGame]['game_image'].values[0]}"

    return render_template('index.html', WhatGame=WhatGame, game_img=game_img)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
