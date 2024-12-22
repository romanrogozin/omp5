#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <iomanip>


std::vector<std::vector<int>> prepare_matrix(int& size, int& bandwidth) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, INT_MAX);

    std::vector<std::vector<int>> matrix(size, std::vector<int>(size, 0));

    for (int i = 0; i < matrix.size(); ++i) {
        // Обрабатываем только элементы в пределах заданной ширины ленты
        for (int j = std::max(0, i - bandwidth); j <= std::min(static_cast<int>(matrix[i].size() - 1), i + bandwidth); ++j) 
        {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}


// Функция для поиска максимального значения среди минимальных элементов строк ленточной матрицы
int find_max_among_row_mins_band_matrix(const std::vector<std::vector<int>>& matrix, int bandwidth) {
    int max_of_mins = std::numeric_limits<int>::min();

    #pragma omp parallel
    {
        int local_max = std::numeric_limits<int>::min();

        // Параллельный поиск минимальных элементов в строках
        #pragma omp for
        for (int i = 0; i < matrix.size(); ++i) {
            int row_min = std::numeric_limits<int>::max();

            // Обрабатываем только элементы в пределах заданной ширины ленты
            for (int j = std::max(0, static_cast<int>(i) - bandwidth); j <= std::min(static_cast<int>(matrix[i].size() - 1), static_cast<int>(i) + bandwidth); ++j) 
            {
                if (matrix[i][j] < row_min) {
                    row_min = matrix[i][j];
                }
            }

            if (row_min > local_max) {
                local_max = row_min;
            }
        }

        // Обновление глобального максимума
        #pragma omp critical
        {
            if (local_max > max_of_mins) {
                max_of_mins = local_max;
            }
        }
    }

    return max_of_mins;
}

int main() {
    std::vector<int> thread_experiments = { 1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024 };
    std::vector<int> matrix_size_experiments = { 1'000, 10'000, 50'000 };
    int runs_count = 5;

    for (int i = 0; i < matrix_size_experiments.size(); i++)
    {
        int matrix_size_experiment = matrix_size_experiments[i];
        int bandwidth = matrix_size_experiment / 2;
        std::vector<std::vector<int>> matrix = prepare_matrix(matrix_size_experiment, bandwidth);

        for (int j = 0; j < thread_experiments.size(); j++)
        {
            int current_thread_experiment = thread_experiments[j];
            omp_set_num_threads(current_thread_experiment);

            double total_execution_time = 0;
            double result = 0;
            for (int k = 0; k < runs_count; k++)
            {
                auto start = std::chrono::high_resolution_clock::now();

                result = find_max_among_row_mins_band_matrix(matrix, bandwidth);

                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> duration = end - start;
                total_execution_time += duration.count();
            }
            double avg_exexution_time = total_execution_time / runs_count;

            std::cout << std::setprecision(10) << matrix_size_experiment << ";" << current_thread_experiment << ";" << avg_exexution_time << std::endl;
        }
    }

    std::cout << "Waiting for exit...";
    int temp;
    std::cin >> temp;
    return 0;
}