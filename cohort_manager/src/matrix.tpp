/**
 * Minimalist implementation of an indexable 2-dimensional array.
 **/

#include <iostream>
#include <utility>
#include <stdexcept>

using std::pair;
using std::size_t;
using std::cout;
using std::endl;

/**
 * Constructor and initializer.
 **/
template <class T>
Matrix<T>::Matrix(size_t m, size_t n, const T& init_value) {
    m_ = m;
    n_ = n;
    data_ = new T[m * n];

    for (size_t i = 0; i < m * n; i++)
        data_[i] = init_value;
}

/**
 * Destructor.
 **/
template <class T>
Matrix<T>::~Matrix() {
    delete data_;
}

/**
 * Set specific elements of the matrix.
 *
 * TODO A nicer version would use the return of the overloaded operator[] to
 * support assignment. This would allow mat[i][j] = 3.
 *
 **/
template <class T>
void Matrix<T>::set(size_t i, size_t j, const T& val) {
    data_[i * n_ + j] = val;
}

/**
 * Get an element using a pair object (instead of indexing).
 **/
template <class T>
T* Matrix<T>::get(pair<size_t, size_t> pos) {
    return &((*this)[pos.first][pos.second]);
}

/**
 * Safe version of the set method.
 **/
template <class T>
void Matrix<T>::safe_set(size_t i, size_t j, const T& val) {
    if (i >= m_ || j >= n_)
        throw std::domain_error("Out of bounds operation on Matrix.");
    set(i, j, val);
}

/**
 * This operator indexes on the row only. After that, the second bracket
 * will index a regular array.
 **/
template <class T>
T* Matrix<T>::operator[](size_t i) { return data_ + i * n_; }

/**
 * Display the matrix.
 *
 * T needs to be printable for this to work.
 **/
template <class T>
void Matrix<T>::print() {
    for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
            cout << (*this)[i][j] << ", ";
        }
        cout << endl;
    }
}
