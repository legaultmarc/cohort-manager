#include <string>
#include <iostream>
#include <algorithm>
#include <cstddef>

#include "includes/query.h"


using std::cout;                using std::endl;
using std::string;              using std::size_t;


// From http://stackoverflow.com/questions/735204/convert-a-string-in-c-to-upper-case
// convert('t') will be 'T'.
struct convert {
    void operator()(char& c) { c = toupper((unsigned char)c); }
};


/**
 * Debug function to print the dynamic programming matrix.
 **/
void printMat(int **mat, const string& query, const string& word) {
    size_t m = word.length() + 1;
    size_t n = query.length() + 1;

    cout << "       ";
    for (size_t j = 0; j < query.length(); j++) {
        cout << query[j];
        if (j != query.length() - 1)
            cout << ", ";
    }
    cout << endl;

    for (size_t i = 0; i < m; i++) {
        if (i > 0) cout << word[i - 1] << " ";
        else if (i == 0) cout << "  ";

        for (size_t j = 0; j < n; j++) {
            if (j == 0) cout << "[ ";
            cout << mat[i][j];
            if (j == n - 1) cout << "] ";
            else cout << ", ";
        }
        cout << endl;
    }
}


/**
 * Return score for the alignment of word in query.
 * word should be a subset of query, so gaps will be allowed on both ends of
 * word.
 **/
double alignScore(const string& query, const string& word) {
    size_t i, j, m, n;
    m = word.length() + 1;
    n = query.length() + 1;
    int **mat;
    mat = new int*[m];

    // Allocate columns.
    for (i = 0; i < m; i++) {
        mat[i] = new int[n];
        if (i == 0) {
            for (j = 0; j < n; j++) mat[i][j] = 0;
        } else {
            mat[i][0] = 0;
        }
    }

    // Fill the matrix.
    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            // Match
            int match = mat[i - 1][j - 1];
            if (word[i - 1] == query[j - 1])
                match += 4;
            else
                match -= 2;

            // Horizontal and vertical indels.
            int h_indel = mat[i - 1][j] - 1;
            int v_indel = mat[i][j - 1] - 1;

            // Choose max.
            mat[i][j] = match;
            if (match > mat[i][j])
                mat[i][j] = match;
            else if (h_indel > mat[i][j])
                mat[i][j] = h_indel;
            else if (v_indel > mat[i][j])
                mat[i][j] = v_indel;
        }
    }

    // The score of the alignment is the max in the last row.
    int max = 0;
    for (j = 0; j < n; j++) {
        if (mat[m - 1][j] > max)
            max = mat[m - 1][j];
    }

    // Deallocate the matrix.
    for (i = 0; i < m; i++) {
        delete mat[i];
    }
    delete mat;


    return max / (4.0 * word.length());
}
