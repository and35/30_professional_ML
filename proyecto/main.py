from utils import Utils
from models import Models
from time import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start_time = time()

    utils = Utils()
    models = Models()
    
    # 1 cargamos datos
    print(" step 1 ... ")
    data = utils.load_from_csv('in/felicidad.csv')
    
    # 2. cargamos variables 
    print(round(time() - start_time, 3), "\n step 2 ...")
    X, y = utils.feature_target(data, ['score', 'rank', 'country'], ['score'])

    """ 3. importacion del mejor modelo
        3.1 definimos 2 modelos (SVG y GBR) y varios configuraciones de parametros a testear con GRID
        3.2 escogemos el modelo con el score mas alto 
        3.3 exportamos el modelo"""
    print(round(time() - start_time, 3), "\n step 3 ...")
    models.grid_training(X, y)
    print(round(time() - start_time, 3), "\n finished !")
