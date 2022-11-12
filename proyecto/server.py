import joblib
import numpy as np

from flask import Flask, app # flas nos permite crear la APIrest
from flask import jsonify # herramienta para trabajar cno arch json

## toda la teoria de esta clase la podras encontrar en el doc despliegue_de_modelos.doc
app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    """Funcion que se expondra en la direccion 8080/predict y que muestra la prediccion hecha
    por nuestro modelo que exportamos al archivo best_model.pkl"""
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1, -1)) 
    return jsonify({'prediccion': list(prediction)})


if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl') 
    app.run(port=8080)