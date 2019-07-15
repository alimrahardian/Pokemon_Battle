from flask import Flask, render_template, request, redirect
import requests
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

combats = pd.read_csv('pokemon/combats.csv')
pokemon = pd.read_csv('pokemon/pokemon.csv')
winlose = ['lose', 'win']

# pre-process
combats['win_label'] = np.where(combats['Winner'] == combats['First_pokemon'], 0, 1)
# print(combats.head())

# feature-target
x = combats.drop(['Winner', 'win_label'], axis=1)
y = combats['win_label']

# TST
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# ml model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
randomFr = RandomForestClassifier(n_estimators=100)
randomFr.fit(xtrain, ytrain)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # try:
        # hp, attack, def, special attack, special def, speed
        url = 'https://pokeapi.co/api/v2/pokemon/{}/'
        cari = request.form['cari1']
        req_pokemon = requests.get(url.format(cari)).json()
        poke_stats = {
            'nama' : req_pokemon['name'],
            'gambar' : req_pokemon['sprites']['front_default'],
            'id' : req_pokemon['id'],
            'HP' : pokemon['HP'].iloc[req_pokemon['id']],
            'Attack' : pokemon['Attack'].iloc[req_pokemon['id']],
            'Defense' : pokemon['Defense'].iloc[req_pokemon['id']],
            'Sp_Atk' : pokemon['Sp. Atk'].iloc[req_pokemon['id']],
            'Sp_Def' : pokemon['Sp. Def'].iloc[req_pokemon['id']],
            'Speed' : pokemon['Speed'].iloc[req_pokemon['id']]
        }

        cari2 = request.form['cari2']
        req_pokemon2 = requests.get(url.format(cari2)).json()
        poke_stats2 = {
            'nama' : req_pokemon2['name'],
            'gambar' : req_pokemon2['sprites']['front_default'],
            'id' : req_pokemon2['id'],
            'HP' : pokemon['HP'].iloc[req_pokemon2['id']],
            'Attack' : pokemon['Attack'].iloc[req_pokemon2['id']],
            'Defense' : pokemon['Defense'].iloc[req_pokemon2['id']],
            'Sp_Atk' : pokemon['Sp. Atk'].iloc[req_pokemon2['id']],
            'Sp_Def' : pokemon['Sp. Def'].iloc[req_pokemon2['id']],
            'Speed' : pokemon['Speed'].iloc[req_pokemon2['id']]
        }

        
        combat_proba = {'a' : randomFr.predict_proba([[poke_stats['id'], poke_stats2['id']]])[0][1]}

        return render_template (
            'result.html', 
            poke_stats=poke_stats, 
            poke_stats2=poke_stats2,
            combat_proba=combat_proba
            )
        # except:
        #     return redirect ('error.html')
    else:
        return render_template ('home.html')


@app.route('/result')
def result():
    return render_template ('result.html')


@app.errorhandler(404)
def notFound404(error):
    return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)