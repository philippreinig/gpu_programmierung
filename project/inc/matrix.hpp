#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <stddef.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>  

#include "util.hpp"

class Matrix
{
public:
    Matrix(uint rows, uint cols) : rows(rows), cols(cols)
    {
        data = new double[rows*cols];
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols){
        data = new double[rows*cols];
        std::memcpy((void*) data, (void*) other.data, sizeof(double)* rows * cols);
    }

    Matrix(uint rows, uint cols, double* data) : rows(rows), cols(cols), data(data){}

    double* operator[](uint index) const
    {
        return &(data[index*this->cols]);
    }

    Matrix operator*(const Matrix &other) const
    {
        if (this->cols != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions don't match");
        }

        Matrix result(this->rows, other.cols);
        for (uint i = 0; i < rows; ++i)
        {
            for (uint j = 0; j < other.cols; ++j)
            {
                for (uint k = 0; k < cols; ++k)
                {
                    result[i][j] += (*this)[i][k] * other[k][j];
                }
            }
        }
        return result;
    }

    double* get_values(){
        return this->data;
    }

    Matrix transpose()
    {
        Matrix result(this->cols, this->rows);
        for (uint i = 0; i < this->rows; ++i)
        {
            for (uint j = 0; j < this->cols; ++j)
            {
                result[j][i] = (*this)[i][j];
            }
        }

        return result;
    }

    void fill_randomly()
    {
        for (uint i = 0; i < rows; ++i)
        {
            for (uint j = 0; j < cols; ++j)
            {
                (*this)[i][j] = rand_double();
            }
        }
    }

    static void fill_randomly(double* M, uint rows, uint cols){
        for (uint i = 0; i < rows; ++i)
        {
            for (uint j = 0; j < cols; ++j)
            {
                M[i*cols+j] = rand_double();
            }
        }
    }

    void fill_ascending()
    {
        unsigned int count = 0;
        for (uint i = 0; i < rows; ++i)
        {
            for (uint j = 0; j < cols; ++j)
            {
                (*this)[i][j] = count;
                ++count;
            }
        }
    }

    void print()
    {
        for (uint i = 0; i < this->rows; ++i)
        {
            for (uint j = 0; j < this->cols; ++j)
            {
                std::cout << (*this)[i][j] << ' ';
            }
            std::cout << '\n';
        }
    }

    static void multiply(Matrix *M_1, Matrix *M_2, Matrix *M_res)
    {
        if (M_1->cols != M_2->rows)
        {
            throw std::invalid_argument("Dimensions don't match!");
        }

        for (uint i = 0; i < M_1->rows; ++i)
        {
            for (uint j = 0; j < M_2->cols; ++j)
            {
                (*M_res)[i][j] = 0;

                for (uint k = 0; k < M_1->cols; ++k)
                {
                    (*M_res)[i][j] += (*M_1)[i][k] * (*M_2)[k][j];
                }
            }
        }
    }

    static void multiply_cache_optimized(Matrix *M_1, Matrix *M_2, Matrix *M_res)
    {
        if (M_1->cols != M_2->rows)
        {
            throw std::invalid_argument("Dimensions don't match!");
        }

        // auto M_2_t = M_2->transpose();

        for (uint i = 0; i < M_1->rows; ++i)
        {
            for (uint k = 0; k < M_1->cols; ++k)
            {
                for (uint j = 0; j < M_2->rows; ++j)
                {
                    (*M_res)[i][j] += (*M_1)[i][k] * (*M_2)[j][k];
                }
            }
        }
    }

    ~Matrix()
    {
        delete data;
    }

    uint rows;
    uint cols;

private:
    double* data;
};

__global__ void multiply_parallel_worker(double* M_1, uint m_1_rows, uint m_1_cols,
                                         double* M_2, uint m_2_rows, uint m_2_cols,
                                         double* M_res);

void multiply_parallel(double* M_1, uint m_1_rows, uint m_1_cols,
                       double* M_2, uint m_2_rows, uint m_2_cols,
                       double* M_res);

#endif