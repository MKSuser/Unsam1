import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

df_limpio = pd.read_csv(r'C:\Users\Ke Ba\Desktop\TP Mate3\df_limpio.csv')

# Extraer variables de entrada (todas las filas, todas las columnas menos de "class")
X = (df_limpio.values[:, :-1])

# Extraer columna de salida (todas las filas, columna "class")
Y = df_limpio.values[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar los datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

#red
nn = MLPClassifier(solver='adam',
                   hidden_layer_sizes=(10, 10, ),
                   activation='relu',
                   max_iter=50_000,
                   learning_rate_init=.001)

#entrenamiento
nn.fit(X_train, Y_train)

print("Porcentaje de aciertos con train: ", (nn.score(X_train, Y_train)*100))
print("Porcentaje de aciertos con test: ", (nn.score(X_test, Y_test)*100))
