#include <string>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <utility>

#include "includes/matrix.h"
#include "includes/query.h"


using std::cout;                using std::endl;
using std::string;              using std::size_t;
using std::pair;


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
Alignment alignScore(const string& query, const string& word) {
    size_t i, j, m, n;
    pair<size_t, size_t> max_coors(0, 0);  // Position of top score.

    m = word.length() + 1;
    n = query.length() + 1;
    Matrix<int> mat(m, n, 0);
    Matrix<Move> moves(m, n, DIAG);

    // Fill the matrix.
    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            // Alignment of characters (i.e. match/mismatch).
            int match = mat[i - 1][j - 1];
            if (word[i - 1] == query[j - 1])
                match += 3;  // Match score
            else {
                match -= 1;  // Mismatch score
            }

            // Horizontal and vertical indels.
            int h_indel = mat[i - 1][j] - 2;
            int v_indel = mat[i][j - 1] - 2;

            // Choose max.
            mat[i][j] = match;
            if (h_indel > mat[i][j]) {
                mat.set(i, j, h_indel);
                moves.set(i, j, UP);
            }
            else if (v_indel > mat[i][j]) {
                mat.set(i, j, v_indel);
                moves.set(i, j, LEFT);
            }

            if (mat[i][j] > *mat.get(max_coors)) {
                max_coors.first = i;
                max_coors.second = j;
            }
        }
    }

    // Walk back the local alignment up to the first zero.
    int max_score = *(mat.get(max_coors));

    // Bounds of the "explained" sub-sequence of the query.
    int left = 0;
    int right = max_coors.second - 1; // -1 because of the column of zeros.

    int cur_score = max_score;
    pair<size_t, size_t> cur(max_coors);
    pair<size_t, size_t> prev(cur);
    while (cur_score != 0) {
        // Go back.
        prev = cur;
        switch (*moves.get(cur)) {
            case UP:
                cur.first--;
                break;
            case LEFT:
                cur.second--;
                break;
            case DIAG:
                cur.first--;
                cur.second--;
                break;
        }
        cur_score = *(mat.get(cur));
    }
    left = prev.second - 1;

    Alignment ret;
    ret.score = max_score;
    ret.left = left;
    ret.right = right;

    return ret;
}
