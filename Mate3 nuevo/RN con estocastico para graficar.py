# Con estocastico / Pruebas para printeo de gráficos

# Llamado de las librerias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

df_limpio = pd.read_csv(r'C:\Users\KB\Desktop\Mate3 Nuevo\df_limpio.csv')

# Declaramos el dataFrame como all_data
all_data = df_limpio

# Separamos los datos a tener en cuenta, por un lado los datos de entrada, por el otro los de salida ("class")
all_inputs = all_data.iloc[:, 0:5].values # Entradas: columnas 0 a 5
all_outputs = all_data.iloc[:, -1].values # Salidas: última columna

# Declaramos la funcion a utilizar para normalizar los datos
def normalizador(X):
    # Calculamos la media y la desviación estándar
    promedio = np.mean(X, axis=0)  # Promedio de cada columna
    desvEst = np.std(X, axis=0)    # Desviación estándar de c/c

    datoNormalizado = (X - promedio) / desvEst
    return datoNormalizado

all_inputs = normalizador(all_inputs)

### Convertir las salidas a formato one-hot

# Declaramos la cantidad de respuestas posibles
# Tener en cuenta que las salidas son 3
respPosibles = 3

# Con nume.eye(respPosibles) generamos una matriz identidad 3x3 que va a reemplazar a los valores (0,1,2) originales
# Al combinarla con all_outputs vinculamos cada fila de la MId con cada valor
# La primer fila sería 0, la segunda 1, y la tercera 2
# Hacemos esto para poder trabajar las 3 posibilidades dentro de la red.
y_matriz = np.eye(respPosibles)[all_outputs]

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, y_matriz, test_size=1/3)

n = X_train.shape[0]  # número de registros de entrenamiento

print(n)

### Funciones de activación

# Función ReLu
relu = lambda x: np.maximum(x, 0)

# Función softmax ------- Agregar como funciona ---------
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def red(L, iters):

    ### Construimos la red neuronal con pesos y sesgos iniciados aleatoriamente

    # Primero determinamos un seed para controlar los valores random

    np.random.seed(16429)

    w_hidden = (np.random.rand(3, 5) * 2) - 1 # Pesos de la capa oculta (3 neuronas, 5 entradas)
    w_output = (np.random.rand(3, 3) * 2) - 1 # Pesos de la capa de salida (3 clases, 3 neuronas ocultas)

    b_hidden = (np.random.rand(3, 1) * 2) - 1 # Biases de la capa oculta (3 neuronas)
    b_output = (np.random.rand(3, 1) * 2) - 1 # Biases de la capa de salida (3 clases)

    # Función del forward para recorrer la red de atrás para adelante
    def forward_prop(X):
        Z1 = w_hidden @ X.T + b_hidden
        A1 = relu(Z1)
        Z2 = w_output @ A1 + b_output
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    conteoTest = []
    conteoTrain = []

    # Cálculo de precisión
    def accuracyTest(X, Y):
        test_predictions = forward_prop(X)[3]  # solo nos interesa A2
        predicted_classes = np.argmax(test_predictions, axis=0)  # obtener clase con mayor probabilidad
        true_classes = np.argmax(Y, axis=1)  # etiquetas verdaderas
        accuracy = np.mean(predicted_classes == true_classes)  # porcentaje de aciertos
        # print("Porcentaje de aciertos: ", (accuracy * 100).round(2))
        conteoTest.append(accuracy)

    def accuracyTrain(X, Y):
        test_predictions = forward_prop(X)[3]  # solo nos interesa A2
        predicted_classes = np.argmax(test_predictions, axis=0)  # obtener clase con mayor probabilidad
        true_classes = np.argmax(Y, axis=1)  # etiquetas verdaderas
        accuracy = np.mean(predicted_classes == true_classes)  # porcentaje de aciertos
        # print("Porcentaje de aciertos: ", (accuracy * 100).round(2))
        conteoTrain.append(accuracy)

        # # Printeamos el accuracy que generamos con el forward
        # print('Pre entrenamiento: \n')
        # print('Test')
        # accuracy(X_test, Y_test)
        # print('Train')
        # accuracy(X_train, Y_train)

    # Devuelve pendientes para pesos y sesgos usando la regla de la cadena
    # Derivada de ReLU
    def d_relu(Z):
        return (Z > 0).astype(float)

    # Derivada de softmax
    def d_softmax(muestra):
        s = muestra.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

              #((3,1),(3,1),(3,1),(3,1),(1,5),(3,1))
    def backward_prop(Z1, A1, Z2, A2, X, Y):
        m = X.shape[0]  # número de ejemplos

        dC_dA2 = A2 - Y.T  # (3, 1)
        dA2_dZ2 = d_softmax(A2)  # (3, 3)
        dZ2_dA1 = w_output.T  # (3, 3)
        dZ2_dW2 = A1.T
        dZ2_dB2 = 1
        dA1_dZ1 = d_relu(Z1)  # (3, 1)
        dZ1_dW1 = X
        dZ1_dB1 = 1

    #            ((3, 1) @ (1, m))  / m = (3, m)
        dC_dW2 = (dC_dA2 @ dZ2_dW2) / m

    #                       ((3, 1) * 1 ) / m
        dC_dB2 = np.sum(dC_dA2, axis=1, keepdims=True) * dZ2_dB2 / m

    #             (3,3)  @  [(3, 3) @ (3, 1) = (3,1)] = (3, 1)
        dC_dA1 = dZ2_dA1 @ np.dot(dA2_dZ2, dC_dA2)

    #             (3, 1) * (3, 1) @ (1, 5) = (3,5)
        dC_dW1 = (dC_dA1 * dA1_dZ1 @ dZ1_dW1) / m

    #                    (3, 1) * (3, 1) * 1 = (3, 1)
        dC_dB1 = np.sum((dC_dA1 * dA1_dZ1 * dZ1_dB1), axis=1, keepdims=True) / m

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

        # Funciones para graficar
        accuracyTest(X_test, Y_test)
        accuracyTrain(X_train, Y_train)

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

iters_l = [1000, 5000, 10000, 50000]
L_l = [0.001, 0.005, 0.01, 0.05]

for iter in iters_l:
    for L in L_l:
        red(L=L, iters=iter)
