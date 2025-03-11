#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define N 512        // Increased number of webpages
#define DAMPING 0.85 // Damping Factor (85% probability of user clicking on a link)
#define ITERATIONS 20 // Number of iterations for convergence
#define NUM_THREADS 8 // Number of threads

int graph[N][N];

// Function to generate a random adjacency matrix
double random_double() {
    return rand() / (double)RAND_MAX;
}

void generate_random_graph() {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            graph[i][j] = (random_double() < 0.1) ? 1 : 0; // 10% probability of having a link
        }
    }
}

// Function to compute outbound links
void compute_outbound_links(int out_links[]) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        out_links[i] = 0;
        for (int j = 0; j < N; j++) {
            if (graph[i][j] == 1) {
                out_links[i]++;
            }
        }
    }
}

// Parallel PageRank computation
void compute_pagerank() {
    double rank[N], new_rank[N];
    int out_links[N];
    double total_time = 0;

    omp_set_num_threads(NUM_THREADS);
    
    generate_random_graph();
    compute_outbound_links(out_links);

    // Initialize ranks equally
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        rank[i] = 1.0 / N;
    }

    for (int run = 0; run < 10; run++) {
        double start_time = omp_get_wtime();

        for (int iter = 0; iter < ITERATIONS; iter++) {
            memset(new_rank, 0, sizeof(new_rank));

            #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
            for (int j = 0; j < N; j++) {
                if (out_links[j] > 0) {
                    for (int i = 0; i < N; i++) {
                        if (graph[j][i] == 1) {
                            #pragma omp atomic
                            new_rank[i] += (DAMPING * rank[j] / out_links[j]);
                        }
                    }
                }
            }

            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int i = 0; i < N; i++) {
                new_rank[i] += (1 - DAMPING) / N;
            }

            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int i = 0; i < N; i++) {
                rank[i] = new_rank[i];
            }
        }

        double end_time = omp_get_wtime();
        double exec_time = end_time - start_time;

        #pragma omp critical
        total_time += exec_time;

        printf("\nExecution Time (Parallel, Run %d): %lf seconds", run + 1, exec_time);
    }

    int index[N];
    for (int i = 0; i < N; i++) {
        index[i] = i;
    }

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (int i = 0; i < N - 1; i++) {
        for (int j = 0; j < N - i - 1; j++) {
            if (rank[j] < rank[j + 1]) {
                double temp = rank[j];
                rank[j] = rank[j + 1];
                rank[j + 1] = temp;

                int tempIdx = index[j];
                index[j] = index[j + 1];
                index[j + 1] = tempIdx;
            }
        }
    }

    printf("\n\nTop 10 Final PageRank Scores (Sorted in Descending Order):\n");
    for (int i = 0; i < 10; i++) {
        printf("Page #%d has score: %lf\n", index[i], rank[i]);
    }

    printf("\nAverage Execution Time (Parallel): %lf seconds\n", total_time / 10);
}

int main() {
    compute_pagerank();
    return 0;
}
