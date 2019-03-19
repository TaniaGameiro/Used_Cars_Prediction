from flask import Flask, render_template, request
from sklearn.externals import joblib
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_price', methods=['POST', 'GET'])
def predict_price():
    # get the parameters
    Year = int(request.form['Year'])
    Make = str(request.form['Make'])
    Model = str(request.form['Model'])
    Transmission = str(request.form['Transmisson'])

 # load the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # generate a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)

    # change the dataframe according to the inputs
    df_prediction.at[0, 'Year'] = Year
    df_prediction.at[0, 'Make_'+str(Make)] = 1.0
    df_prediction.at[0, 'Model'] = Model
    df_prediction.at[0, 'Transmission_'+str(Transmisson)] = 1.0
    print(df_prediction)

    # load the model and predict
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_time = prediction.round(1)[0]


return render_template('results.html',
                           Year=int(Year),
                           Make=str(Make),
                           Model=str(Model),
                           Transmission=str(Transmission),
                           predicted_price=int(predicted_time)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
