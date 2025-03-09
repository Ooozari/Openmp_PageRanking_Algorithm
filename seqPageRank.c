#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 16        // Number of webpages
#define DAMPING 0.85 // Damping Factor (85% probability of user clicking on a link)
#define ITERATIONS 20 // Number of iterations for convergence

// Adjacency matrix (links between webpages)
// 0 means no link 1 mean have link
int graph[N][N] = {
    {0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0},
    {1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0},
    {1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1},
    {0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1},
    {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0},
    {0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1},
    {0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0},
    {0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1},
    {1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0},
    {0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1},
    {0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0},
    {1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1},
    {0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1},
    {1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0},
    {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1},
    {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0}
};

// Function to compute outbound links for each page
void compute_outbound_links(int out_links[]) {
    for (int i = 0; i < N; i++) {
        out_links[i] = 0;
        for (int j = 0; j < N; j++) {
            if (graph[i][j] == 1) {
                out_links[i]++;  // Counting number of links from page i
            }
        }
    }
}

// Function to compute PageRank
void compute_pagerank() {
    double rank[N], new_rank[N]; 
    int out_links[N];
    clock_t start, end;
    double singleT;
    double total_time = 0;

    // Initialize ranks equally 
    for (int i = 0; i < N; i++) {
        rank[i] = 1.0 / N;
    }

    compute_outbound_links(out_links);

    // Running 10 times for average execution time analysis
    for (int run = 0; run < 10; run++) {
        start = clock();

        // PageRank iteration process
        for (int iter = 0; iter < ITERATIONS; iter++) { 
            for (int i = 0; i < N; i++) {
                new_rank[i] = (1 - DAMPING) / N;
            }

            // Summing Up the contributions from all linked pages
            for (int j = 0; j < N; j++) {
                if (out_links[j] > 0) { 
                    for (int i = 0; i < N; i++) {
                        if (graph[j][i] == 1) {
                            new_rank[i] += (DAMPING * rank[j] / out_links[j]);
                        }
                    }
                }
            }

            // Updating rank array with calculated rank
            for (int i = 0; i < N; i++) {
                rank[i] = new_rank[i];
            }
        }

        end = clock();
        singleT = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("\nExecution Time (Sequential): %lf seconds", singleT);
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    // Sorting PageRank scores in descending order
    int index[N];
    for (int i = 0; i < N; i++) {
        index[i] = i; // Store original index of pages
    }

    // Bubble Sort (Descending Order)
    for (int i = 0; i < N - 1; i++) {
        for (int j = 0; j < N - i - 1; j++) {
            if (rank[j] < rank[j + 1]) {
                // Swap values in rank array
                double temp = rank[j];
                rank[j] = rank[j + 1];
                rank[j + 1] = temp;

                // Swap corresponding indexes
                int tempIdx = index[j];
                index[j] = index[j + 1];
                index[j + 1] = tempIdx;
            }
        }
    }

    // Displaying Final PageRank Scores in Descending Order
    printf("\n\nFinal PageRank Scores (Sorted in Descending Order):\n");
    for (int i = 0; i < N; i++) {
        printf("Page %d: %lf\n", index[i], rank[i]);
    }

    printf("\nAverage Execution Time (Sequential): %f seconds\n", total_time / 10);
}

int main() {
    compute_pagerank();
    return 0;
}