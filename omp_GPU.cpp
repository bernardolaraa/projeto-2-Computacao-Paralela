#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;

class Point {
private:
    int id_point, id_cluster;
    vector<double> values;
    int total_values;
    string name;

public:
    Point(int id_point, vector<double> &values, string name = "") {
        this->id_point = id_point;
        total_values = values.size();
        this->values = values;
        this->name = name;
        id_cluster = -1;
    }

    int getID() const {
        return id_point;
    }

    void setCluster(int id_cluster) {
        this->id_cluster = id_cluster;
    }

    int getCluster() const {
        return id_cluster;
    }

    double getValue(int index) const {
        return values[index];
    }

    int getTotalValues() const {
        return total_values;
    }

    string getName() const {
        return name;
    }
};

class KMeans {
private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;

public:
    KMeans(int K, int total_points, int total_values, int max_iterations) {
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
    }

    void run(vector<Point> &points) {
        if (K > total_points)
            return;

        // Convert points to array
        double *points_values = new double[total_points * total_values];
        int *points_cluster = new int[total_points];

        for (int i = 0; i < total_points; i++) {
            points_cluster[i] = points[i].getCluster();
            for (int j = 0; j < total_values; j++) {
                points_values[i * total_values + j] = points[i].getValue(j);
            }
        }

        // Initialize centroids
        double *centroids = new double[K * total_values];
        int *prohibited_indexes = new int[K];

        srand(time(NULL));

        for (int i = 0; i < K; i++) {
            while (true) {
                int index_point = rand() % total_points;

                bool found = false;
                for (int j = 0; j < i; j++) {
                    if (prohibited_indexes[j] == index_point) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    prohibited_indexes[i] = index_point;
                    points_cluster[index_point] = i;
                    for (int j = 0; j < total_values; j++) {
                        centroids[i * total_values + j] = points_values[index_point * total_values + j];
                    }
                    break;
                }
            }
        }

        delete[] prohibited_indexes;

        int iter = 1;

        while (true) {
            int done = 1;

// Associar cada ponto ao centro mais próximo - paralelizado na GPU
#pragma omp target data map(to: points_values[0:total_points * total_values], centroids[0:K * total_values]) \
                        map(tofrom: points_cluster[0:total_points]) map(tofrom: done)
            {
#pragma omp target teams distribute parallel for reduction(&:done)
                for (int i = 0; i < total_points; i++) {
                    int id_old_cluster = points_cluster[i];
                    int id_nearest_center = getIDNearestCenter(points_values, centroids, i);

                    if (id_old_cluster != id_nearest_center) {
                        done = 0;
                        points_cluster[i] = id_nearest_center;
                    }
                }
            }

            // Recalcular o centro de cada cluster
            double *new_centroids = new double[K * total_values];
            int *clusters_size = new int[K];

            fill(new_centroids, new_centroids + K * total_values, 0.0);
            fill(clusters_size, clusters_size + K, 0);

#pragma omp target data map(to: points_values[0:total_points * total_values], points_cluster[0:total_points]) \
                        map(tofrom: new_centroids[0:K * total_values], clusters_size[0:K])
            {
#pragma omp target teams distribute parallel for
                for (int i = 0; i < total_points; i++) {
                    int cluster_id = points_cluster[i];
#pragma omp atomic
                    clusters_size[cluster_id]++;

                    for (int j = 0; j < total_values; j++) {
#pragma omp atomic
                        new_centroids[cluster_id * total_values + j] += points_values[i * total_values + j];
                    }
                }
            }

            // Atualizar centroids
            for (int i = 0; i < K; i++) {
                if (clusters_size[i] > 0) {
                    for (int j = 0; j < total_values; j++) {
                        centroids[i * total_values + j] = new_centroids[i * total_values + j] / clusters_size[i];
                    }
                }
            }

            delete[] new_centroids;
            delete[] clusters_size;

            if (done == 1 || iter >= max_iterations) {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }

            iter++;
        }

        // Atualizar pontos com as atribuições finais do cluster
        for (int i = 0; i < total_points; i++) {
            points[i].setCluster(points_cluster[i]);
        }

        delete[] points_values;
        delete[] points_cluster;
        delete[] centroids;
    }

    int getIDNearestCenter(double *points_values, double *centroids, int point_index) {
        double min_dist = 1e10;
        int id_cluster_center = 0;

        for (int i = 0; i < K; i++) {
            double dist = 0.0;
            for (int j = 0; j < total_values; j++) {
                double diff = centroids[i * total_values + j] - points_values[point_index * total_values + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                id_cluster_center = i;
            }
        }

        return id_cluster_center;
    }
};

int main(int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    int total_points, total_values, K, max_iterations, has_name;

    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    for (int i = 0; i < total_points; i++) {
        vector<double> values;

        for (int j = 0; j < total_values; j++) {
            double value;
            cin >> value;
            values.push_back(value);
        }

        if (has_name) {
            cin >> point_name;
            Point p(i, values, point_name);
            points.push_back(p);
        } else {
            Point p(i, values);
            points.push_back(p);
        }
    }

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Tempo de execução: " << elapsed.count() << " segundos\n";

    return 0;
}