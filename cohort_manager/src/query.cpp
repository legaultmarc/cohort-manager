#include <string>
#include <iostream>
#include <algorithm>
#include <cstddef>

#include "includes/matrix.h"
#include "includes/query.h"


using std::cout;                using std::endl;
using std::string;              using std::size_t;


// From http://stackoverflow.com/questions/735204/convert-a-string-in-c-to-upper-case
// convert('t') will be 'T'.
struct convert {
    void operator()(char& c) { c = toupper((unsigned char)c); }
};


/**
 * Return score for the alignment of word in query.
 * word should be a subset of query, so gaps will be allowed on both ends of
 * word.
 **/
double alignScore(const string& query, const string& word) {
    size_t i, j, m, n;
    m = word.length() + 1;
    n = query.length() + 1;
    Matrix<int> mat(m, n, 0);

    // Fill the matrix.
    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            // Alignment of characters (i.e. match/mismatch).
            int match = mat[i - 1][j - 1];
            if (word[i - 1] == query[j - 1])
                match += 2;  // Match score
            else
                match -= 1;  // Mismatch score

            // Horizontal and vertical indels.
            int h_indel = mat[i - 1][j] - 1;
            int v_indel = mat[i][j - 1] - 1;

            // Choose max.
            mat[i][j] = match;
            if (match > mat[i][j])
                mat.set(i, j, match);
            else if (h_indel > mat[i][j])
                mat.set(i, j, h_indel);
            else if (v_indel > mat[i][j])
                mat.set(i, j, v_indel);
        }
    }

    // The score of the alignment is the max in the last row.
    int max = 0;
    for (j = 0; j < n; j++) {
        if (mat[m - 1][j] > max)
            max = mat[m - 1][j];
    }

    return max;
}
