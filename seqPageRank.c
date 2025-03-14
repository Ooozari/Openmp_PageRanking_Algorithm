#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512        // Number of webpages
#define DAMPING 0.85 // Damping Factor (85% probability of user clicking on a link)
#define ITERATIONS 20 // Number of iterations for convergence

int graph[N][N]; // Adjacency matrix

// Function to generate a random graph with ~10% links
void generate_random_graph() {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                graph[i][j] = (rand() % 100) < 10 ? 1 : 0;
            } else {
                graph[i][j] = 0;
            }
        }
    }
}

// Compute outbound links for each page
void compute_outbound_links(int out_links[]) {
    for (int i = 0; i < N; i++) {
        out_links[i] = 0;
        for (int j = 0; j < N; j++) {
            if (graph[i][j] == 1) {
                out_links[i]++;
            }
        }
    }
}

// Compute PageRank sequentially
void compute_pagerank() {
    double rank[N], new_rank[N];
    int out_links[N];
    clock_t start, end;
    double total_time = 0;

    // Initialize ranks equally
    for (int i = 0; i < N; i++) {
        rank[i] = 1.0 / N;
    }

    compute_outbound_links(out_links);

    for (int run = 0; run < 10; run++) {
        start = clock();

        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (int i = 0; i < N; i++) {
                new_rank[i] = (1 - DAMPING) / N;
            }

            for (int j = 0; j < N; j++) {
                if (out_links[j] > 0) {
                    for (int i = 0; i < N; i++) {
                        if (graph[j][i] == 1) {
                            new_rank[i] += (DAMPING * rank[j] / out_links[j]);
                        }
                    }
                }
            }

            for (int i = 0; i < N; i++) {
                rank[i] = new_rank[i];
            }
        }

        end = clock();
        double singleT = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Execution Time (Sequential): %lf seconds\n", singleT);
        total_time += singleT;
    }

    // Sorting PageRank scores in descending order
    int index[N];
    for (int i = 0; i < N; i++) {
        index[i] = i;
    }

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

    printf("\nTop 10 Final PageRank Scores (Sorted in Descending Order):\n");
    for (int i = 0; i < 10; i++) {
        printf("Page # %d has score: %lf\n", index[i], rank[i]);
    }

    printf("\nAverage Execution Time (Sequential): %f seconds\n", total_time / 10);
}

int main() {
    generate_random_graph();
    compute_pagerank();
    return 0;
}
