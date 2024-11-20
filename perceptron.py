import random
from sklearn.datasets import load_iris

# Función para cargar y filtrar el dataset Iris
def load_iris_binary():
    iris = load_iris()
    data = []
    for features, label in zip(iris.data, iris.target):
        # Filtrar solo Setosa (0) y Virginica (2)
        if label in [0, 2]:
            binary_label = 0 if label == 0 else 1
            data.append(list(features) + [binary_label])
    random.shuffle(data)  # Mezclar datos
    return data

# Dividir datos en entrenamiento y prueba
def train_test_split(data, test_ratio=0.3):
    split_index = int(len(data) * (1 - test_ratio))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

# Función de activación del perceptrón (Paso escalón)
def step_function(value):
    return 1 if value >= 0 else 0

# Implementación del Perceptrón Simple
def perceptron_train(train_data, learning_rate=0.1, epochs=100):
    num_features = len(train_data[0]) - 1  # Excluir etiqueta
    weights = [random.uniform(-1, 1) for _ in range(num_features)]
    bias = random.uniform(-1, 1)

    for _ in range(epochs):
        for sample in train_data:
            features = sample[:-1]
            label = sample[-1]
            prediction = step_function(sum(w * x for w, x in zip(weights, features)) + bias)
            error = label - prediction

            # Actualización de pesos y bias
            weights = [w + learning_rate * error * x for w, x in zip(weights, features)]
            bias += learning_rate * error

    return weights, bias

# Predicción con el perceptrón entrenado
def perceptron_predict(weights, bias, features):
    return step_function(sum(w * x for w, x in zip(weights, features)) + bias)

# Evaluar exactitud del perceptrón
def evaluate(test_data, weights, bias):
    correct = 0
    total = len(test_data)
    for sample in test_data:
        features = sample[:-1]
        label = sample[-1]
        prediction = perceptron_predict(weights, bias, features)
        if prediction == label:
            correct += 1
    accuracy = correct / total
    return accuracy

# Main
if __name__ == "__main__":
    # Cargar datos desde sklearn y dividir en entrenamiento/prueba
    data = load_iris_binary()
    train_data, test_data = train_test_split(data)

    # Entrenar el perceptrón
    weights, bias = perceptron_train(train_data)

    # Evaluar el desempeño
    accuracy = evaluate(test_data, weights, bias)

    # Resultados
    print("Pesos finales:", weights)
    print("Bias final:", bias)
    print("Exactitud en prueba (Hold-Out):", accuracy)
