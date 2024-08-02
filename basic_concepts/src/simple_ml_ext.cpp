#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>
namespace py = pybind11;

void print_matrix(float ** matrix,int max_row,int max_col){
    for(int row=0;row<max_row;row++)
    {
        std::cout<<"[ ";
        for(int col=0;col<max_col;col++)
        {
            std::cout<<matrix[row][col]<<" ";
        }
        std::cout<<"] \n";
    }
}

void print_fakematrix(const float * matrix,int max_row,int max_col){
    for(int row=0;row<max_row;row++)
    {
        std::cout<<"[ ";
        for(int col=0;col<max_col;col++)
        {
            std::cout<<matrix[row*max_col+col]<<" ";
        }
        std::cout<<"] \n";
    }
}
void print_fakematrix1(const unsigned char * matrix,int max_row,int max_col){
    for(int row=0;row<max_row;row++)
    {
        std::cout<<"[ ";
        for(int col=0;col<max_col;col++)
        {
            std::cout<<int(matrix[row*max_col+col])<<" ";
        }
        std::cout<<"] \n";
    }
}

// generated by GPT4o, I FINISH one version, but met unsolved segmentation fault
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
     // Allocate memory for logits and gradients
    std::vector<float> logits(batch * k, 0.0f);
    std::vector<float> gradients(n * k, 0.0f);

    for (size_t start = 0; start < m; start += batch) {
        size_t end = std::min(start + batch, m);
        size_t current_batch_size = end - start;

        // Reset logits and gradients for the current batch
        std::fill(logits.begin(), logits.end(), 0.0f);
        std::fill(gradients.begin(), gradients.end(), 0.0f);

        // Compute logits for the current batch
        for (size_t i = 0; i < current_batch_size; ++i) {
            for (size_t j = 0; j < k; ++j) {
                for (size_t l = 0; l < n; ++l) {
                    logits[i * k + j] += X[(start + i) * n + l] * theta[l * k + j];
                }
            }
        }

        // Apply softmax to logits
        for (size_t i = 0; i < current_batch_size; ++i) {
            float max_logit = *std::max_element(logits.begin() + i * k, logits.begin() + (i + 1) * k);
            float sum_exp = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                logits[i * k + j] = std::exp(logits[i * k + j] - max_logit);
                sum_exp += logits[i * k + j];
            }
            for (size_t j = 0; j < k; ++j) {
                logits[i * k + j] /= sum_exp;
            }
        }

        // Compute gradients
        for (size_t i = 0; i < current_batch_size; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float label = (y[start + i] == j) ? 1.0f : 0.0f;
                float diff = logits[i * k + j] - label;
                for (size_t l = 0; l < n; ++l) {
                    gradients[l * k + j] += diff * X[(start + i) * n + l];
                }
            }
        }

        // Update theta
        for (size_t l = 0; l < n; ++l) {
            for (size_t j = 0; j < k; ++j) {
                theta[l * k + j] -= lr * gradients[l * k + j] / current_batch_size;
            }
        }
    }
    /// END YOUR CODE

}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
