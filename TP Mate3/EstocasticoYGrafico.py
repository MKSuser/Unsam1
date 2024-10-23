# LLamado de las librerias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Declaramos el dataFrame como all_data
all_data = pd.read_csv(r'C:\Users\Ke Ba\Desktop\TP Mate3\df_limpio.csv')

# Separamos los datos a tener en cuenta, por un lado los datos de entrada, por el otro los de salida ("class")
all_inputs = all_data.iloc[:, 0:8].values
all_outputs = all_data.iloc[:, -1].values  # Asegúrate de que esta columna tenga 3 clases (0, 1, 2)

# Declaramos la funcion a utilizar para normalizar los datos y le damos play
def normalizador(X):
    # Calcular la media y la desviación estándar
    promedio = np.mean(X, axis=0)  # Media de cada característica
    desvEst = np.std(X, axis=0)    # Desviación estándar de cada característica

    # Aplicar la transformación
    datoNormalizado = (X - promedio) / desvEst
    return datoNormalizado

all_inputs = normalizador(all_inputs)

### Convertir las salidas a formato one-hot
respPosibles = 3
y_matriz = np.eye(respPosibles)[all_outputs]

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, y_matriz, test_size=1/3)

n = X_train.shape[0]  # número de registros de entrenamiento

# Funciones de activación
relu = lambda x: np.maximum(x, 0)

# Función softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # estabilización numérica
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def red(L, iters):

    # Construir una red neuronal con pesos y sesgos iniciados aleatoriamente
    np.random.seed(1)
    w_hidden = (np.random.rand(10, 8) * 2) - 1
    w_output = (np.random.rand(3, 10) * 2) - 1

    b_hidden = (np.random.rand(10, 1) * 2) - 1
    b_output = (np.random.rand(3, 1) * 2) - 1

    # Función que corre la red neuronal con los datos de entrada para predecir la salida
    def forward_prop(X):
        Z1 = w_hidden @ X.T + b_hidden
        A1 = relu(Z1)
        Z2 = w_output @ A1 + b_output
        A2 = softmax(Z2)  # Cambiamos a softmax para la salida
        return Z1, A1, Z2, A2

    conteoTest = []
    conteoTrain = []

    # Cálculo de precisión
    def precisionTest(X, Y):
        test_predictions = forward_prop(X)[3]  # solo nos interesa A2
        predicted_classes = np.argmax(test_predictions, axis=0)  # obtener clase con mayor probabilidad
        true_classes = np.argmax(Y, axis=1)  # etiquetas verdaderas
        accuracy = np.mean(predicted_classes == true_classes)  # porcentaje de aciertos
        # print("Porcentaje de aciertos: ", (accuracy * 100).round(2))
        conteoTest.append(accuracy)

    def precisionTrain(X, Y):
        test_predictions = forward_prop(X)[3]  # solo nos interesa A2
        predicted_classes = np.argmax(test_predictions, axis=0)  # obtener clase con mayor probabilidad
        true_classes = np.argmax(Y, axis=1)  # etiquetas verdaderas
        accuracy = np.mean(predicted_classes == true_classes)  # porcentaje de aciertos
        # print("Porcentaje de aciertos: ", (accuracy * 100).round(2))
        conteoTrain.append(accuracy)

    #print('Pre entrenamiento: \n')
    #print('Test')
    #precisionTest(X_test, Y_test)
    #print('Train')
    #precisionTrain(X_train, Y_train)

    # Devuelve pendientes para pesos y sesgos usando la regla de la cadena
    def backward_prop(Z1, A1, Z2, A2, X, Y):
        m = X.shape[0]  # número de ejemplos
        dC_dA2 = A2 - Y.T  # error de salida
        dC_dW2 = (dC_dA2 @ A1.T) / m  # (3, m) @ (m, 10) = (3, 10)
        dC_dB2 = np.sum(dC_dA2, axis=1, keepdims=True) / m  # (3, m) -> (3, 1)

        dA1_dZ1 = (Z1 > 0).astype(float)  # derivada de ReLU
        dC_dA1 = w_output.T @ dC_dA2  # (10, 3) @ (3, m) = (10, m)
        dC_dZ1 = dC_dA1 * dA1_dZ1  # (10, m)

        dC_dW1 = (dC_dZ1 @ X) / m  # (10, m) @ (m, 8) = (10, 8)
        dC_dB1 = np.sum(dC_dZ1, axis=1, keepdims=True) / m  # (10, m) -> (10, 1)

        return dC_dW1, dC_dB1, dC_dW2, dC_dB2

    # La tasa de aprendizaje
    # L = 0.001

    # Ejecutar descenso de gradiente estocástico
    # num_epochs = 1_000  # Aumentar el número de épocas

    for i in tqdm(range(iters)):
        idx = np.random.randint(0, n)  # Elegir un solo índice aleatorio

        X_sample = X_train[idx:idx+1]  # Obtener el ejemplo
        Y_sample = Y_train[idx:idx+1]  # Obtener la etiqueta correspondiente

        Z1, A1, Z2, A2 = forward_prop(X_sample)

        dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

        w_hidden -= L * dW1
        b_hidden -= L * dB1
        w_output -= L * dW2
        b_output -= L * dB2

        #Cálculo de precisión
        #print('Post entrenamiento: \n')
        #print('Test')
        precisionTest(X_test, Y_test)
        #print('Train')
        precisionTrain(X_train, Y_train)

    graficar_accuracy(L=L, train_accuracies = conteoTrain, test_accuracies=conteoTest)

import matplotlib.pyplot as plt

def graficar_accuracy(L, train_accuracies, test_accuracies):

    iters = len(test_accuracies)

    fmt_train = {
        'color': 'tab:blue',
        'ls': 'solid',
        'lw': 3,
    }

    fmt_test = {
        'color': 'tab:orange',
        'ls': 'solid',
        'lw': 3,
    }

    fig, (ax) = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(train_accuracies, label='Train', **fmt_train)
    ax.plot(test_accuracies, label='Test', **fmt_test)

    ax.grid(which='both')
    ax.legend()
    ax.set_title(f'Accuracy {L=}')
    ax.set_xlabel('Step')

    fig.tight_layout()
    plt.savefig(f'accuracy_{L=}_{iters=}.png')
    plt.show()

iters_l = [75000]
L_l = [0.001]

for iter in iters_l:
    for L in L_l:
        red(L=L, iters=iter)