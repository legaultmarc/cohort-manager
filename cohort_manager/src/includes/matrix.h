#ifndef GUARD_ml_matrix_h
#define GUARD_ml_matrix_h

#include <utility>
#include <cstddef>
/**
 * Matrix class.
 **/
template <class T> class Matrix {
    public:
        Matrix(size_t m, size_t n, const T& init_value = T());
        ~Matrix();
        T* operator[](size_t i);
        void set(size_t, size_t, const T&);
        T* get(std::pair<size_t, size_t>);
        void safe_set(size_t, size_t, const T&);
        void print();

    private:
        T* data_;
        size_t m_;
        size_t n_;
};

#include "../matrix.tpp"

#endif
