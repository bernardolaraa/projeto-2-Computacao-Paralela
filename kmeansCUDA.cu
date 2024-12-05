#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <omp.h> // Incluído para o OpenMP

using namespace std;

// Função __device__ fora da classe
__device__ float computeDistance(const float *point_values, const float *center_values, int total_values) {
    float sum = 0.0f;
    for (int i = 0; i < total_values; i++) {
        float diff = point_values[i] - center_values[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Funções __global__ fora da classe
__global__ void assignClusters(
    const float *points, const float *centers, int *cluster_assignments,
    int total_points, int total_values, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_points) {
        const float *point = &points[idx * total_values];
        float min_dist = 1e10f;
        int nearest_center = 0;

        for (int k = 0; k < K; k++) {
            const float *center = &centers[k * total_values];
            float dist = computeDistance(point, center, total_values);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_center = k;
            }
        }
        cluster_assignments[idx] = nearest_center;
    }
}

__global__ void updateCenters(
    const float *points, const int *cluster_assignments, float *new_centers,
    int *points_per_cluster, int total_points, int total_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_points) {
        int cluster = cluster_assignments[idx];
        atomicAdd(&points_per_cluster[cluster], 1);

        for (int j = 0; j < total_values; j++) {
            atomicAdd(&new_centers[cluster * total_values + j], points[idx * total_values + j]);
        }
    }
}

__global__ void normalizeCenters(
    float *centers, const int *points_per_cluster, int total_values, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        int num_points = points_per_cluster[idx];
        if (num_points > 0) {
            for (int j = 0; j < total_values; j++) {
                centers[idx * total_values + j] /= num_points;
            }
        }
    }
}

class Point {
private:
    int id_point, id_cluster;
    vector<float> values; // Usando 'float' em vez de 'double'
    int total_values;
    string name;

public:
    Point(int id_point, vector<float> &values, string name = "")
        : id_point(id_point), id_cluster(-1), values(values), total_values(values.size()), name(name) {}

    int getID() { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() { return id_cluster; }
    float getValue(int index) { return values[index]; }
    int getTotalValues() { return total_values; }
    string getName() { return name; }
};

class KMeans {
private:
    int K, total_values, total_points, max_iterations;

public:
    KMeans(int K, int total_points, int total_values, int max_iterations)
        : K(K), total_points(total_points), total_values(total_values), max_iterations(max_iterations) {}

    void run(vector<Point> &points) {
        if (K > total_points) return;

        vector<float> centers(K * total_values, 0.0f);
        vector<int> cluster_assignments(total_points, -1);

        // Inicializa aleatoriamente os centros dos clusters
        srand(time(nullptr));
        for (int i = 0; i < K; i++) {
            int idx = rand() % total_points;
            for (int j = 0; j < total_values; j++) {
                centers[i * total_values + j] = points[idx].getValue(j);
            }
        }

        // Aloca memória na GPU
        float *d_points, *d_centers, *d_new_centers;
        int *d_cluster_assignments, *d_points_per_cluster;

        cudaMalloc(&d_points, total_points * total_values * sizeof(float));
        cudaMalloc(&d_centers, K * total_values * sizeof(float));
        cudaMalloc(&d_new_centers, K * total_values * sizeof(float));
        cudaMalloc(&d_cluster_assignments, total_points * sizeof(int));
        cudaMalloc(&d_points_per_cluster, K * sizeof(int));

        // Copia os pontos para a GPU
        vector<float> flattened_points(total_points * total_values);
        for (int i = 0; i < total_points; i++) {
            for (int j = 0; j < total_values; j++) {
                flattened_points[i * total_values + j] = points[i].getValue(j);
            }
        }
        cudaMemcpy(d_points, flattened_points.data(), total_points * total_values * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (total_points + blockSize - 1) / blockSize;

        for (int iter = 0; iter < max_iterations; iter++) {
            // Copia os centros atuais para a GPU
            cudaMemcpy(d_centers, centers.data(), K * total_values * sizeof(float), cudaMemcpyHostToDevice);

            // Atribui pontos aos clusters
            assignClusters<<<numBlocks, blockSize>>>(d_points, d_centers, d_cluster_assignments, total_points, total_values, K);

            // Reinicia os novos centros e o contador de pontos por cluster
            cudaMemset(d_new_centers, 0, K * total_values * sizeof(float));
            cudaMemset(d_points_per_cluster, 0, K * sizeof(int));

            // Atualiza os centros dos clusters
            updateCenters<<<numBlocks, blockSize>>>(d_points, d_cluster_assignments, d_new_centers, d_points_per_cluster, total_points, total_values);

            // Normaliza os centros
            int numBlocksCenters = (K + blockSize - 1) / blockSize;
            normalizeCenters<<<numBlocksCenters, blockSize>>>(d_new_centers, d_points_per_cluster, total_values, K);

            // Copia os novos centros de volta para a CPU
            cudaMemcpy(centers.data(), d_new_centers, K * total_values * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(cluster_assignments.data(), d_cluster_assignments, total_points * sizeof(int), cudaMemcpyDeviceToHost);

        // Libera a memória da GPU
        cudaFree(d_points);
        cudaFree(d_centers);
        cudaFree(d_new_centers);
        cudaFree(d_cluster_assignments);
        cudaFree(d_points_per_cluster);

        // Você pode adicionar código aqui para processar os resultados, se desejar.
    }
};

int main(int argc, char *argv[]) {
    srand(0);
    auto start = std::chrono::high_resolution_clock::now();

    int num_threads = 1;
    if (argc > 2) {
        num_threads = atoi(argv[1]);
    } else {
        cout << "Uso: ./kmeansCUDA <num_threads> <input_file>" << endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    // Abre o arquivo
    std::ifstream input_file(argv[2]);
    if (!input_file.is_open()) {
        cout << "Erro ao abrir o arquivo: " << argv[2] << endl;
        return 1;
    }

    int total_points, total_values, K, max_iterations, has_name;
    input_file >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    for (int i = 0; i < total_points; i++) {
        vector<float> values;

        for (int j = 0; j < total_values; j++) {
            float value;
            input_file >> value;
            values.push_back(value);
        }

        if (has_name) {
            input_file >> point_name;
            Point p(i, values, point_name);
            points.push_back(p);
        } else {
            Point p(i, values);
            points.push_back(p);
        }
    }

    input_file.close();

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Tempo de execução com " << num_threads << " thread(s): " << elapsed.count() << " segundos\n";

    return 0;
}