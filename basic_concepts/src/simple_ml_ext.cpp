#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>
namespace py = pybind11;


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

    // X (m,n)  theta (n,k)
    for(int start=0 ; start < m ; start = start + batch ):
        int end=min(start+batch,m)

        // calculate the Xtheta using for loop
        // Xtheta's shape is (m,k)
        float ** Xtheta = new float *[m];
        for(int index=0;index<m;index++){
            Xtheta[index]=new float[k];
        }

        for(int i=0;i+start<end;i++)
        {
            float temp=0.0;
            for(int p=0;p<k;p++)
            {    
                for(int j=0;j<n;j++)
                {            
                        temp+=X[i*m+j] * theta[j*n+p];
                }
            }
            Xtheta[i][p]=temp;
        }

            //calculate max logits
            // Xtheta's shape = (m,k)
        float * max_logits=new float[m];
        for(int i=0 ; i<m ; i++)
        {
            float max_number=Xtheta[i][0];
            for (int j=0 ; j <k ; j++)
            {   
                if(Xtheta[i][j]>max_number)
                    max_number=Xtheta[i][j];
            }
            max_logits[i]=max_number;
        }

        // Xtheta-max_logits and exp
        for(int i=0 ; i<m ; i++)
        {
            for (int j=0 ; j <k ; j++)
            {
                Xtheta[i][j]=exp(Xtheta[i][j]-max_logits[i]);
            }
        }

        // calculate Z (refer simple_ml.py)
        float * exp_sum = new float [m];
        for(int i=0 ; i<m ; i++)
        {
            float sum_num=0.0;
            for (int j=0 ; j <k ; j++)
            {   
                sum_num+=Xtheta[i][j]
            }
            exp_sum[i]=sum_num;
        }

        // use Xtheta memmory for Z
        for(int i=0 ; i<m ; i++)
        {
            for (int j=0 ; j <k ; j++)
            {
                Xtheta[i][j]=Xtheta[i][j]/exp_sum[i];
            }
        }
        
        // Ey shape is (batch, k)
        float ** Ey =new float * [batch];
        for(int index=0 ; index<batch; index++){
            Ey[index]=new float[k];
            for(int j=0;j<k;j++)
            {
                Ey[index][j]=0;
            }
        }

        // generate one-hot code

        for(int row=0;row+start<end;row++){
            Ey[row][y[row+start]]=1
        }

        float ** gradients= new float *[n];
        for(int i=0;i<n;i++){
            gradients[i]=new float[k];
        }
        //  gradients's shape is (n,k)
        for(int col=0; col< n; col++)
        {
            float temp=0.0;
            for(int next_col=0 ; next_col < k ; next_col++) {
                
                for(int row=0; row+start < end ; row++)
                {
                    // gradients[col][next_col]=X[row*n+col]*(Xtheta[row*k+col]-
                    temp+=X[row*n+col]*(Xtheta[row*k+next_col]-Ey[row*k+next_col])
                }
                gradients[col][next_col]=temp/batch;
            }
        }

        // upgrade theta
        // theta shape (n,k)
        for(int row=0;row<n;row++)
        {
            for(int col=0;col<k;col++)
            {
                theta[row][col]-= lr * gradients[row][col];
            }
        }


        // release the memory
        // float ** Xtheta = new float *[m];
        // float * max_logits=new float[m];
        // float * exp_sum = new float [m];
        // float ** Ey =new float * [batch];
        // float ** gradients= new float *[n];
        for(int index=0;index<m;index++){
            delete Xtheta[index];
        }
        delete Xtheta;
        delete max_logits;
        delete exp_sum;

        for(int index=0;index<m;index++){
            delete Ey[index];
        }
        delet Ey;

        for(int index=0;index<m;index++){
            delete gradients[index];
        }
        delet gradients;

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
