import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df_limpio = pd.read_csv(r'C:\Users\KB\Desktop\Mate3 Nuevo\df_limpio.csv')

# Extraer variables de entrada (todas las filas, todas las columnas menos de "class")
X = (df_limpio.values[:, :-1])

# Extraer columna de salida (todas las filas, columna "class")
Y = df_limpio.values[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar los datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

#red
nn = MLPClassifier(solver='sgd',
                   hidden_layer_sizes=(3, ),
                   activation='relu',
                   max_iter=50_000,
                   learning_rate_init=.005)

#entrenamiento
nn.fit(X_train, Y_train)

print("Porcentaje de aciertos con train: ", (nn.score(X_train, Y_train)*100))
print("Porcentaje de aciertos con test: ", (nn.score(X_test, Y_test)*100))
