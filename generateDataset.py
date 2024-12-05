import random


def generate_large_dataset(filename, num_points):
    with open(filename, "w") as file:
        # Cabeçalho: total de pontos, número de valores, número de clusters, máximo de iterações, e flag de nome
        file.write(f"{num_points} 4 3 100 1\n")

        # Gerar pontos Iris-setosa
        for _ in range(num_points // 3):
            values = [
                random.uniform(4.3, 5.8),
                random.uniform(2.3, 4.4),
                random.uniform(1.0, 1.9),
                random.uniform(0.1, 0.6),
            ]
            name = "Iris-setosa"
            file.write(" ".join(map(str, values)) + " " + name + "\n")

        # Gerar pontos Iris-versicolor
        for _ in range(num_points // 3):
            values = [
                random.uniform(4.9, 7.0),
                random.uniform(2.0, 3.4),
                random.uniform(3.0, 5.1),
                random.uniform(1.0, 1.8),
            ]
            name = "Iris-versicolor"
            file.write(" ".join(map(str, values)) + " " + name + "\n")

        # Gerar pontos Iris-virginica
        for _ in range(num_points // 3):
            values = [
                random.uniform(5.8, 7.9),
                random.uniform(2.5, 3.8),
                random.uniform(4.5, 6.9),
                random.uniform(1.4, 2.5),
            ]
            name = "Iris-virginica"
            file.write(" ".join(map(str, values)) + " " + name + "\n")


# Especificar o número de pontos e o nome do arquivo
num_points = 1000000  # Número grande de pontos
filename = "largest_dataset.txt"

# Gerar o dataset
generate_large_dataset(filename, num_points)
print(f"Dataset com {num_points} pontos gerado no arquivo {filename}.")
