/// CMSC714 (Fall2019, High Performance Computing Systems)
/// Group Project: Paris (PARallel Iterative Solver)
///     ... distributed parallel (MPI+OpenMP hybrid) implementation of GMRES
/// Members: Shoken Kaneko, Jiaxing Liang, Yi Zhou
/// Implemented in C++17.

/// Complilation:
///   The latest code depends on Eigen and CUDA (CUDA is not yet used though). 
///   Build it with the Makefile to include/link to those libraries by running:
/// >> module load gcc/9.1.0
/// >> module load openmpi/gnu/9.1.0/4.0.1
/// >> module load eigen/3.3.5
/// >> module load cuda/9.1.85
/// >> make

/// Obsolete:
/// without MPI,
/// >> g++ paris.cpp -std=c++17 -fopenmp -O3
/// with MPI,
/// >> mpic++ paris.cpp -std=c++17 -fopenmp -O3

/// Interface (current):
/// >> mpirun -n <#processes> ./a.out <#threadsPerProcess> <decompositionType> <matrixName> <tolerance> <dense/sparse> <algorithm> <#maxIterations>
/// execution example (current):
/// >> mpirun -n 12 ./a.out 1 2r sherman4 1e-10 sparse cgs -1

/// Interface (meant for future. not yet implemented like this):
/// >> mpirun -n <#processes> ./a.out <#threadsPerProcess> <decompositionType> <matrixName> <tolerance> <dense/sparse> <algorithm> <#maxIterations> <real/complex> <float/double>

/// decompositionType has to be chosen from {2r, 2c, 1r, 1c}.
/// matrixName has to be chosen from {sherman1, sherman2, sherman3, sherman4, sherman5}.
/// algorithm has to be chosen from {cgs, cgs_l1}.
/// if maxIteration is set to -1, it will be set to the dimension of the matrix.
/// warning: currently, maxIteration is fixed to the matrix dimension.
/// warning: currently, the path to the matrix and vector is hard-coded. put the "matrices" directory next to the executable.

// TODO: measure: GMRES-spmv performance
// TODO: measure: study decomposition effect in the sparse case
// TODO: measure: Eigen version (dense, serial / parallel)
// TODO: analyze: do performance analysis using hpcToolkit or tau or other tools
// TODO: analyze: compare with GMRES implementations of trilinos / hypre
// TODO: develop: optimize by reducing communication and MPI_Barriers that are not required
// TODO: develop: implement algorithm variants from Yamazaki2017
// TODO: develop: try load balancing of spmv
// TODO: develop: use GPU

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <memory>
#include <cmath>
#include <complex>
#include <functional>
#include <omp.h>
#include <cassert>

#define INCLUDE_MPI
#ifdef INCLUDE_MPI
#include "mpi.h"
# ifndef MPI_INCLUDED
# define MPI_INCLUDED
#endif
#endif

//#define USE_EIGEN
#ifdef USE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

using namespace std;

auto getTime() {
#ifdef MPI_INCLUDED
	auto t = MPI_Wtime();
#else
	auto t = omp_get_wtime();
#endif
	return t;
}

template <class T>
T* allocVec(const int size_x) {
	T* array = new T[size_x];
	// initialize to zero.
	for (int i = 0; i < size_x; i++) {
		array[i] = 0;
	}
	return array;
}

template <class T>
void deleteVec(T* array) {
	delete[] array;
}

template <class T>
T** allocMat(const int size_x, const int size_y) {
	T* p;
	T** A = new T* [size_x];
	p = new T[size_x * size_y];
	for (int i = 0; i < size_x; i++) {
		A[i] = &p[i * size_y];
	}
	// initialize to zero.
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			A[i][j] = 0;
		}
	}
	return A;
}

template <class T>
void deleteMat(T** array){
	T* p = array[0];
	delete[] p;
	delete[] array;
}

template <class T>
tuple<T*, int*, int*> allocSpMat(const int n, const int nnz) {

	int* csrRowPtr = new int [n+1];
	int* csrColInd = new int [nnz];
	T* csrVal = new T[nnz];

	for (int i = 0; i < n+1; i++) {
		csrRowPtr[i] = 0;
	}

	for (int i = 0; i < nnz; i++) {
		csrColInd[i] = 0;
		csrVal[i] = 0;
	}

	return {csrVal, csrColInd, csrRowPtr};
}

template <class T>
void deleteSpMat(T* csrVal, int* csrColInd, int* csrRowPtr){
	delete[] csrVal;
	delete[] csrColInd;
	delete[] csrRowPtr;
}

complex<double> myconj(complex<double> c) {
	return conj(c);
}
double myconj(double c) {
	return c;
}
complex<float> myconj(complex<float> c) {
	return conj(c);
}
float myconj(float c) {
	return c;
}

#ifdef USE_EIGEN
template<class T, int ndim>
auto alloc_eigen(const vector<int>& shape) {
	if constexpr (ndim == 2) {
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(shape[0], shape[1]);
		mat.setZero();
		return mat;
	}
	else if constexpr (ndim == 1) {
		Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::RowMajor> mat(shape[0], 1);
		mat.setZero();
		return mat;
	}
}
#endif

/* dense mat vec product */
template <class T>
void dense_mat_vec(T* result, const T** A, const T* v, const int H, const int W) {
	//#pragma omp target
	{
		#pragma omp parallel for
		//#pragma acc kernels
		//#pragma acc loop independent
		for (int i = 0; i < H; i++) { // no dependency over i
			result[i] = 0;
			for (int k = 0; k < W; k++) {
				result[i] += A[i][k] * v[k];
			}
		}
	}
}

/* sparse mat vec product */
template <class T>
void sparse_mat_vec(T* result, T* csrVal, int* csrColInd, int* csrRowPtr, T* v, int H){
	// H: number of rows in result vector.
	#pragma omp parallel for
	for (int i = 0; i < H; ++i) {
		result[i] = 0;
		for (int k = csrRowPtr[i]; k < csrRowPtr[i+1]; ++k) {
			result[i] += csrVal[k] * v[csrColInd[k]];
		}
	}
}

/* dense mat vec product */
// ref: https://simulationcorner.net/index.php?page=fastmatrixvector
template <class T>
void dense_mat_vec_with_blocking(T* result,  T** A, T* v, const int H, const int W) {
	T ytemp;
	T* Apos = &(A[0][0]);
	for (int i = 0; i < H; i++)
	{
		T* xpos = &(v[0]);
		ytemp = 0;
		for (int j = 0; j < W; j++)
		{
			ytemp += (*(Apos++)) * (*(xpos++));
		}
		result[i] = ytemp;
	}
}


/* dense mat vec product with blocking */
// ref: https://simulationcorner.net/index.php?page=fastmatrixvector
template <class T>
void dense_mat_vec_with_blocking_2(T* result, T** A, T* v, const int H, const int W) {
	T* Apos1 = &A[0][0];
	T* Apos2 = &A[1][0];
	T* ypos = &result[0];
	for (int i = 0; i < H / 2; i++)
	{
		T ytemp1 = 0;
		T ytemp2 = 0;
		T* xpos = &v[0];
		for (int j = 0; j < W; j++)
		{
			ytemp1 += (*(Apos1++)) * (*xpos);
			ytemp2 += (*(Apos2++)) * (*xpos);
			xpos++;
		}
		*ypos = ytemp1;
		ypos++;
		*ypos = ytemp2;
		ypos++;
		Apos1 += W;
		Apos2 += W;
	}
}


/* dense mat vec product with blocking */
// ref: https://simulationcorner.net/index.php?page=fastmatrixvector
template <class T>
void dense_mat_vec_with_blocking_3(T* result, T** A, T* v, const int H, const int W) {
	T* Apos1 = &A[0][0];
	T* Apos2 = &A[1][0];
	T* Apos3 = &A[2][0];
	T* Apos4 = &A[3][0];
	T* ypos = &result[0];
	for (int i = 0; i < H / 4; i++)
	{
		T ytemp1 = 0;
		T ytemp2 = 0;
		T ytemp3 = 0;
		T ytemp4 = 0;
		T* xpos = &v[0];
		for (int j = 0; j < W; j++)
		{
			ytemp1 += (*(Apos1++)) * (*xpos);
			ytemp2 += (*(Apos2++)) * (*xpos);
			ytemp3 += (*(Apos3++)) * (*xpos);
			ytemp4 += (*(Apos4++)) * (*xpos);
			xpos++;
		}
		*ypos = ytemp1;
		ypos++;
		*ypos = ytemp2;
		ypos++;
		*ypos = ytemp3;
		ypos++;
		*ypos = ytemp4;
		ypos++;
		Apos1 += 3 * W;
		Apos2 += 3 * W;
		Apos3 += 3 * W;
		Apos4 += 3 * W;
	}
}




/*dense vec vec product*/
template <class T>
T dense_vec_vec(const T* v1, const T* v2, const int N) {
	T result = 0;
	for (int i = 0; i < N; i++) {
		result += v1[i] * v2[i];
	}
	return result;
}

/*dense vec.conj().dot(vec) product*/
template <class T>
T dense_vecconj_vec(const T* v1, const T* v2, const int N) {
	T result = 0;
	if constexpr (is_same_v<T, complex<double>> || is_same_v<T, complex<float>>) {
		//#pragma omp parallel reduction(+:result)
		for (int i = 0; i < N; i++) {
			result += conj(v1[i]) * v2[i];
		}
	}
	else {
		//#pragma omp parallel reduction(+:result)
		for (int i = 0; i < N; i++) {
			result += v1[i] * v2[i];
		}
	}
	return result;
}

/* computes 2-norm of a vector */
template <class T>
double norm(const T* vec, const int len) {
	auto n2 = dense_vecconj_vec<T>(vec, vec, len);
	return abs(sqrt(n2));
}

template <class T>
void mult_factor_inplace(T* v, const T fac, const int N) {
	for (int i = 0; i < N; i++) {
		v[i] *= fac;
	}
}

template <class T>
void add_vector_inplace(T* v, const T* v1, const int N){
	for (int i = 0; i < N; i++) {
		v[i] += v1[i];
	}
}

/*could be optimized further*/
template <class T>
void calc_residual(T* result, const T** A, const T* b, const T* x0, const int N) {
	dense_mat_vec<T>(result, A, x0, N, N); // r0 == A.x0
	mult_factor_inplace<T>(result, -1, N); // r0 == -A.x0
	add_vector_inplace<T>(result, b, N); // r0 == b-A.x0
}

template <class T>
void set_column(T** A, const T* col, const int colIdx, const int nRows, const int nCols) {
	for (int i = 0; i < nRows; i++) {
		A[i][colIdx] = col[i];
	}
}

template <class T>
void set_row(T** M, const T* row, const int rowIdx, const int nRows, const int nCols) {
	for (int i = 0; i < nCols; i++) {
		M[rowIdx][i] = row[i];
	}
}

template <class T>
void mult_factor_to_row(T** M, const T factor, const int rowIdx, const int nRows, const int nCols) {
	for (int i = 0; i < nCols; i++) {
		M[rowIdx][i] *= factor;
	}
}

/* NOTE: this fnc switches behaviour depending on type. if complex, the factor is conjugated. */
template <class T>
void axpy(T* result, const T factor, const T* x, T* y, const int N) {
    if constexpr (is_same_v<T, complex<double>> || is_same_v<T, complex<float>>) {// complex case
		//#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            result[i] = conj(factor) * x[i] + y[i];
        }
    }else{
		//#pragma omp parallel for
		for (int i = 0; i < N; i++) {
            result[i] = factor * x[i] + y[i];
        }
    }
}

/* given a and b, this fnc calculates cs and sn of a 2x2 Givens roration mat and stores them into cs[idx] and sn[idx].*/
// Givens mat G = [[cs  -conj(sn)]
//                 [sn  -conj(cs)]]
// makes G*[a b] = [x 0] where x is some number.
template <class T>
void givens(T a, T b, T* cs, T* sn, int idx) {
	if constexpr (is_same_v<T, complex<double>> || is_same_v<T, complex<float>>) { // complex case
		auto r = sqrt(conj(a) * a + conj(b) * b);
		if (abs(r) == 0) {
			cs[idx] = 1.0;
			sn[idx] = 0.0;
		}
		else {
			cs[idx] = conj(a) / r;
			sn[idx] = -conj(b) / r;
		}
	}
	else { // real case
		auto r = sqrt(a * a + b * b);
		if (abs(r) == 0) {
			cs[idx] = 1.0;
			sn[idx] = 0.0;
		}
		else {
			cs[idx] = a / r;
			sn[idx] = -b / r;
		}
	}
}

/*	triangular solve. solves Rx=b. R must be upper triangular.
	note that this fnc does overwrite R and b to save memory.
    result will be written to b. */
template<class T>
void triangularSolve_inplace_upper(T** R, T* b, const int N) {
	for (int i = 0; i < N; i++) {
		b[i] /= R[i][i];
	}
	for (int i = 0; i < N; i++) {
		auto Ri = R[i];
		auto Rii = Ri[i];
		for (int j = i; j < N; j++) {
			Ri[j] /= Rii;
		}
	}
	for (int n = N - 2; n >= 0; n--) {
		for (int j = n + 1; j < N; j++) {
			b[n] -= R[n][j] * b[j];
		}
	}
}


/*	triangular solve. solves Lx=b. L must be lower triangular.
	note that this fnc does overwrite L and b to save memory.
	result will be written to b. */
template<class T>
void triangularSolve_inplace_lower(T** L, T* b, const int N) {
	for (int i = 0; i < N; i++) {
		b[i] /= L[i][i];
	}
	for (int i = 0; i < N; i++) {
		auto Li = L[i];
		auto Lii = Li[i];
		for (int j = 0; j <= i; j++) {
			Li[j] /= Lii;
		}
	}
	for (int n = 0; n < N; n++) {
		for (int j = 1; j < n; j++) {
			b[n] -= L[n][j] * b[j];
		}
	}
}

template <class T>
void applyGivens(T** Ht, T* cs, T* sn, T* e1, const int k){
    for (int i = 1; i < k; i++) {
        auto tmp = cs[i - 1] * Ht[k - 1][i - 1] - myconj(sn[i - 1]) * Ht[k-1][i];
        Ht[k-1][i] = sn[i - 1] * Ht[k - 1][i - 1] + myconj(cs[i - 1]) * Ht[k-1][i];
        Ht[k - 1][i - 1] = tmp;
    }
    givens<T>(Ht[k - 1][k - 1], Ht[k-1][k], cs, sn, k - 1);
    Ht[k - 1][k - 1] = cs[k - 1] * Ht[k - 1][k - 1] - myconj(sn[k - 1]) * Ht[k-1][k];
    Ht[k-1][k] = 0;
    e1[k] = sn[k - 1] * e1[k - 1];
    e1[k - 1] = cs[k - 1] * e1[k - 1];
}

#ifdef USE_EIGEN
template <class T>
void applyGivens_eigen(Eigen::Matrix<T,-1,-1,Eigen::RowMajor>& Ht, T* cs, T* sn, T* e1, const int k) {
	for (int i = 1; i < k; i++) {
		auto tmp = cs[i - 1] * Ht(k - 1,i - 1) - myconj(sn[i - 1]) * Ht(k - 1,i);
		Ht(k - 1,i) = sn[i - 1] * Ht(k - 1,i - 1) + myconj(cs[i - 1]) * Ht(k - 1,i);
		Ht(k - 1,i - 1) = tmp;
	}
	givens<T>(Ht(k - 1,k - 1), Ht(k - 1,k), cs, sn, k - 1);
	Ht(k - 1,k - 1) = cs[k - 1] * Ht(k - 1,k - 1) - myconj(sn[k - 1]) * Ht(k - 1,k);
	Ht(k - 1,k) = 0;
	e1[k] = sn[k - 1] * e1[k - 1];
	e1[k - 1] = cs[k - 1] * e1[k - 1];
}
#endif
// given the number of domains, create domain decomposition in 2D, where nDomain_x >= nDomain_y
tuple<int, int> getNumDomains_x_y_parallel_2D_row(int nProcess) {
	if (nProcess == 1) {
		return { 1,1 };
	}
	int k = int(log2(nProcess));
	int ret_0 = int(pow(2, k / 2));
	int ret_1 = nProcess / ret_0;
	if (ret_0 >= ret_1)
		return { ret_0, ret_1 };
	else {
		return { ret_1, ret_0 };
	}
}

// given the number of domains, create domain decomposition in 2D, where nDomain_x <= nDomain_y
tuple<int, int> getNumDomains_x_y_parallel_2D_col(int nProcess) {
	auto [a, b] = getNumDomains_x_y_parallel_2D_row(nProcess);
	return { b, a };
}

// given the number of processes, create domain decomposition in 1D
tuple<int, int> getNumDomains_x_y_parallel_1D_row(int nProcess) {
	return { nProcess, 1 };
}
tuple<int, int> getNumDomains_x_y_parallel_1D_col(int nProcess) {
	return { 1, nProcess };
}

int min(int a, int b) {
	return a < b ? a : b;
}
int max(int a, int b) {
	return a > b ? a : b;
}
// get position and size data of a domain
tuple<int, int, int, int, int, int> getDomainRange(const int H, const int W, const int domainIdx_x, const int domainIdx_y, int nDomains_x, int nDomains_y) {
	int rowIdx_x_head_d = (H * domainIdx_x) / nDomains_x;
	int rowIdx_x_tail_d = min((H * (domainIdx_x + 1)) / nDomains_x - 1, H - 1);
	int colIdx_y_head_d = (W * domainIdx_y) / nDomains_y;
	int colIdx_y_tail_d = min((W * (domainIdx_y + 1)) / nDomains_y - 1, W - 1);
	int dH = rowIdx_x_tail_d - rowIdx_x_head_d + 1;
	int dW = colIdx_y_tail_d - colIdx_y_head_d + 1;
	return { rowIdx_x_head_d, rowIdx_x_tail_d, colIdx_y_head_d, colIdx_y_tail_d, dH, dW };
}
// get domainIdx from domainIdx_x and domainIdx_y
int getDomainIdx_from_domainIdx_x_y(int dIdx_x, int dIdx_y, int nDomains_x, int nDomains_y) {
	return dIdx_x * nDomains_y + dIdx_y;
}
// get domainIdx_x and domainIdx_y from domainIdx
tuple<int, int> getDomainIndices_x_y_from_domainIdx(int domainIdx, int nDomains_x, int nDomains_y) {
	int dIdx_x = domainIdx / nDomains_y;
	int dIdx_y = domainIdx - dIdx_x * nDomains_y;
	return { dIdx_x, dIdx_y };
}


// copy matrix.
template <class T>
void copyMat(T** mat_dst, const T** mat_src, const int rowHeadIdx_dst, const int rowHeadIdx_src, const int colHeadIdx_dst, const int colHeadIdx_src, const int H, const int W) {
	for (int i = 0; i < H; i++) {
		T* row_dst = mat_dst[i + rowHeadIdx_dst];
		const T* row_src = mat_src[i + rowHeadIdx_src];
		for (int j = 0; j < W; j++) {
			row_dst[j + colHeadIdx_dst] = row_src[j + colHeadIdx_src];
		}
	}
}

template<class T>
void copyVec(T* dst, const T* src, const int headIdx_dst, const int headIdx_src, const int len) {
	for (int i = 0; i < len; i++) {
		dst[headIdx_dst + i] = src[headIdx_src + i];
	}
}

template<class T>
void copyCol_from_mat_to_vec(T* dst, const T** src, const int headIdx_dst, const int headIdx_src, const int len, const int colIdx) {
	for (int i = 0; i < len; i++) {
		dst[headIdx_dst + i] = src[headIdx_src + i][colIdx];
	}
}

template<class T>
void copyRow_from_mat_to_vec(T* dst, const T** src, const int headIdx_dst, const int headIdx_src, const int len, const int rowIdx) {
	int* src_row = src[rowIdx];
	for (int i = 0; i < len; i++) {
		dst[headIdx_dst + i] = src_row[headIdx_src + i];
	}
}

template<class T>
void copyCol_from_vec_to_mat(T** dst, const T* src, const int headIdx_dst, const int headIdx_src, const int len, const int colIdx) {
	for (int i = 0; i < len; i++) {
		dst[headIdx_dst + i][colIdx] = src[headIdx_src + i];
	}
}

template<class T>
void copyRow_from_vec_to_mat(T** dst, const T* src, const int headIdx_dst, const int headIdx_src, const int len, const int rowIdx) {
	for (int i = 0; i < len; i++) {
		dst[rowIdx][headIdx_dst + i] = src[headIdx_src + i];
	}
}

template<class T>
void printMat_to_stdOut_dense(const T** mat, const int H, const int W) {
	cout << "A:" << endl;
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			cout << mat[i][j] << " ";
		}
		cout << "\n";
	}
	cout << "\n";
}

template<class T>
void printVec_to_stdOut_dense(const T* vec, const int H, const string name) {
	cout << name << endl;
	for (int i = 0; i < H; i++) {
		cout << vec[i] << " " << endl;
	}
	cout << "\n";
}

template<class T>
tuple<T*, int*, int*, int, int> denseToSparse(T** densMat, int H, int W) {
	int nnz0 = 0;
	int nnz = 0;

	// count nonzeros
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			if (densMat[i][j] != 0) {
				nnz0 += 1;
			}
		}
	}

	auto [csrVal, csrColInd, csrRowPtr] = allocSpMat<T>(H, nnz0);

	csrRowPtr[0] = 0;
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			if (densMat[i][j] != 0) {
				csrVal[nnz] = densMat[i][j];
				csrColInd[nnz] = j;
				nnz += 1;
			}
		}
		csrRowPtr[i + 1] = nnz;
	}

	return { csrVal, csrColInd, csrRowPtr, H, nnz };
}

/* this version stores the Hessenberg in a transposed form. should be more efficient. */
template <class T>
int gmres(
	const T** A,
	const T* b,
	T* x0/*initial guess. usually a zero vector. result will be stored here*/,
	int kmax/*max iteration*/,
	const double tol/*relative tolerance*/,
	const int N/*matrix size*/,
	const string& algo,
	const int assumeSparse
	) {

	if (kmax <= 0) {
		kmax = N;
	}

	T** V = allocMat<T>(kmax + 1, N); // Krylov bases
	T** Ht = allocMat<T>(kmax, kmax + 1); // transpose of the Hessenberg matrix
	T* cs = allocVec<T>(kmax); // cosines of the Givens rotation metrices
	T* sn = allocVec<T>(kmax); // sines of the Givens rotation metrices
	T* e1 = allocVec<T>(kmax + 1); // beta*e1 vec which will receive rotations

	calc_residual<T>(V[0], (const T * *)A, (const T*)b, (const T*)x0, N); // r0 == b - A.dot(x0)
	double beta = norm<T>(V[0], N);
	mult_factor_to_row<T>(V, 1.0 / beta, 0, kmax + 1, N); // V[0] == r0/|r0|
	e1[0] = static_cast<T>(beta);
	double resnorm = beta;
	double resnorm0 = beta;
	int k = 1;

	T* csrVal = 0;
	int* csrColInd = 0;
	int* csrRowPtr = 0;
	int H = N;
	int nnz = N*N;
	tuple<T*, int*, int*, int, int> spMat;
	if (assumeSparse) {
		spMat = denseToSparse(const_cast<T**>(A), N, N);
		csrVal = get<0>(spMat);
		csrColInd = get<1>(spMat);
		csrRowPtr = get<2>(spMat);
		H = get<3>(spMat);
		nnz = get<4>(spMat);
	}

	while (resnorm > tol * resnorm0 && k <= kmax) {

		auto Ht_km1 = Ht[k - 1];
		auto Vk = V[k];

		// Arnoldi start. 
		// this is the time consuming part which has to be distributed and parallelized.
		if (!assumeSparse) {
			dense_mat_vec<T>(Vk, A, V[k - 1], N, N);
		}
		else {
			sparse_mat_vec<T>(Vk, csrVal, csrColInd, csrRowPtr, V[k - 1], N);
		}

		double h_kp1k = 0;

		if (algo=="cgs" || algo=="CGS") { // CGS
            for (int i = 1; i < k + 1; i++) { // no dependency over i
                Ht_km1[i - 1] = dense_vecconj_vec<T>(Vk, V[i - 1], N);
            }
            for (int i = 1; i < k + 1; i++) { // no dependency over i
                axpy<T>(Vk, -Ht_km1[i - 1], V[i - 1], Vk, N); // V[k] = -conj(H[i,k-1]) * V[i] + V[k]
            }
			h_kp1k = norm<T>(Vk, N);
        }
		else if (algo == "l1" || algo=="cgs_l1") {
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				Ht_km1[i - 1] = dense_vecconj_vec<T>(Vk, V[i - 1], N); // global reduction here, ...
			}
			h_kp1k = dense_vecconj_vec(Vk, Vk, N); // ... and here.
			h_kp1k -= dense_vecconj_vec(Ht_km1, Ht_km1, k);
			h_kp1k = sqrt(abs(h_kp1k));
			for (int i = 1; i < k + 1; i++) { // no dependency over i
                axpy<T>(Vk, -Ht_km1[i - 1], V[i - 1], Vk, N); // V[k] = -conj(H[i,k-1]) * V[i] + V[k]
            }
		}
		else if (algo=="mgs" || algo=="MGS") { // MGS
            for (int i = 1; i < k + 1; i++) {
                Ht_km1[i - 1] = dense_vecconj_vec<T>(Vk, V[i - 1], N);
                axpy<T>(Vk, -Ht_km1[i - 1], V[i - 1], Vk, N); // V[k] = -conj(H[i-1,k-1]) * V[i-1] + V[k]
            }
			h_kp1k = norm<T>(Vk, N);
        }else{
		    cout << "Specified algorithm is not implemented." << endl;
		    break;
		}

		if (h_kp1k == 0) {
			break; // lucky breakdown
		}
		Ht_km1[k] = h_kp1k;
		mult_factor_to_row<T>(V, 1.0 / h_kp1k, k, kmax + 1, N);


		// Arnoldi done.
        applyGivens<T>(Ht, cs, sn, e1, k);
		resnorm = abs(e1[k]);
		cout << "relResNorm after iter " << k << ": " << resnorm / resnorm0 << endl;
		k++;
	}

	// solve triangular
	triangularSolve_inplace_lower<T>(Ht, e1, k - 1);

	// write resulting x to x0
	for (int i = 0; i < k - 1; i++) {
		axpy(x0, myconj(e1[i]), V[i], x0, N);
	}

	deleteMat<T>(V);// , kmax + 1);
	deleteMat<T>(Ht);// , kmax);
	deleteVec<T>(cs);
	deleteVec<T>(sn);
	deleteVec<T>(e1);

	return k - 1;
}



#ifdef USE_EIGEN
/* this version stores the Hessenberg in a transposed form. should be more efficient. */
template <class T>
int gmres_eigen(
	T** A,
	T* b,
	T* x0/*initial guess. usually a zero vector. result will be stored here*/,
	int kmax/*max iteration*/,
	const double tol/*relative tolerance*/,
	const int N/*matrix size*/,
	const string& algo,
	const int assumeSparse
) {

	if (kmax <= 0) {
		kmax = N;
	}
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > A_eigen(A[0], N, N);
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > b_eigen(b, N, 1);
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > x0_eigen(x0, N, 1);

//	T** V = allocMat<T>(kmax + 1, N); // Krylov bases
	//T** Ht = allocMat<T>(kmax, kmax + 1); // transpose of the Hessenberg matrix
	T* cs = allocVec<T>(kmax); // cosines of the Givens rotation metrices
	T* sn = allocVec<T>(kmax); // sines of the Givens rotation metrices
	T* e1 = allocVec<T>(kmax + 1); // beta*e1 vec which will receive rotations
	auto V = alloc_eigen<T, 2>({ kmax + 1,N });
	auto Ht = alloc_eigen<T, 2>({ kmax, kmax+1 });
	//auto cs = alloc_eigen<T, 2>({ kmax, kmax + 1 });

	//calc_residual<T>(V[0], (const T**)A, (const T*)b, (const T*)x0, N); // r0 == b - A.dot(x0)
	V.row(0) = (b_eigen - A_eigen * x0_eigen).transpose();
	
	//double beta = norm<T>(V[0], N);
	double beta = V.row(0).norm();
	//mult_factor_to_row<T>(V, 1.0 / beta, 0, kmax + 1, N); // V[0] == r0/|r0|
	V.row(0) *= 1.0 / beta;
	e1[0] = static_cast<T>(beta);
	double resnorm = beta;
	double resnorm0 = beta;
	int k = 1;

	Eigen::SparseMatrix<T, Eigen::RowMajor> A_eigen_sp(N, N);
	if (assumeSparse) {
		A_eigen_sp = A_eigen.sparseView().eval();
	}


	while (resnorm > tol* resnorm0&& k <= kmax) {

		auto Ht_km1 = Ht.row(k - 1);
		//auto Vk = V[k];

		// Arnoldi start. 
		// this is the time consuming part which has to be distributed and parallelized.
		if (!assumeSparse) {
			//dense_mat_vec<T>(V[k], A, V[k - 1], N, N);
			V.row(k) = (A_eigen * V.row(k - 1).transpose()).transpose();
		}
		else {
			V.row(k) = (A_eigen_sp * V.row(k - 1).transpose()).transpose();
		}

		if (algo == "cgs" || algo == "CGS") { // CGS
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				//Ht_km1[i - 1] = dense_vecconj_vec<T>(V[k], V[i - 1], N);
				//Ht_km1(i - 1) = V.row(k).conjugate().dot(V.row(i - 1)); // wrong! Eigen's dot is conjugate-linear.
				Ht_km1(i - 1) = V.row(k).dot(V.row(i - 1));
			}
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				//axpy<T>(V[k], -Ht_km1[i - 1], V[i - 1], V[k], N); // V[k] = -conj(H[i,k-1]) * V[i] + V[k]
				V.row(k) = -myconj(Ht_km1(i - 1)) * V.row(i - 1) + V.row(k);
			}
		}
		else if (algo == "mgs" || algo == "MGS") { // MGS
			for (int i = 1; i < k + 1; i++) {
				//Ht_km1[i - 1] = dense_vecconj_vec<T>(V[k], V[i - 1], N);
				//axpy<T>(V[k], -Ht_km1[i - 1], V[i - 1], V[k], N); // V[k] = -conj(H[i-1,k-1]) * V[i-1] + V[k]
				Ht_km1(i - 1) = V.row(k).dot(V.row(i - 1));
				V.row(k) = -myconj(Ht_km1(i - 1)) * V.row(i - 1) + V.row(k);
			}
		}
		else {
			cout << "Specified algorithm is not implemented." << endl;
			break;
		}

		//double h_kp1k = norm<T>(Vk, N);
		double h_kp1k = V.row(k).norm();
		if (h_kp1k == 0) {
			break; // lucky breakdown
		}
		else {
			//Ht_km1[k] = h_kp1k;
			//mult_factor_to_row<T>(V, 1.0 / h_kp1k, k, kmax + 1, N);
			Ht_km1(k) = h_kp1k;
			V.row(k) *= 1.0 / h_kp1k;
		}
		// Arnoldi done.
		applyGivens_eigen<T>(Ht, cs, sn, e1, k);
		resnorm = abs(e1[k]);
		cout << "relResNorm after iter " << k << ": " << resnorm / resnorm0 << endl;
		k++;
	}

	// solve triangular
	//triangularSolve_inplace_lower<T>(Ht, e1, k - 1);

	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > e1_eigen(e1, kmax+1, 1);
	e1_eigen.head(k-1) = Ht.block(0,0,k-1,k-1).colPivHouseholderQr().solve(e1_eigen.head(k-1));


	// write resulting x to x0
	for (int i = 0; i < k - 1; i++) {
		//axpy(x0, myconj(e1[i]), V[i], x0, N);
		x0_eigen = myconj(e1_eigen(i)) * V.row(i).transpose() + x0_eigen;
	}

	//deleteMat<T>(V);// , kmax + 1);
	//deleteMat<T>(Ht);// , kmax);
	deleteVec<T>(cs);
	deleteVec<T>(sn);
	deleteVec<T>(e1);

	return k - 1;
}



/* this version stores the Hessenberg in a transposed form. should be more efficient. */
template <class T>
int gmres_eigen_ver2(
	T** A,
	T* b,
	T* x0/*initial guess. usually a zero vector. result will be stored here*/,
	int kmax/*max iteration*/,
	const double tol/*relative tolerance*/,
	const int N/*matrix size*/,
	const string& algo,
	const int assumeSparse
) {

	if (kmax <= 0) {
		kmax = N;
	}
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > A_eigen(A[0], N, N);
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > b_eigen(b, N, 1);
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > x0_eigen(x0, N, 1);

	//	T** V = allocMat<T>(kmax + 1, N); // Krylov bases
		//T** Ht = allocMat<T>(kmax, kmax + 1); // transpose of the Hessenberg matrix
	T* cs = allocVec<T>(kmax); // cosines of the Givens rotation metrices
	T* sn = allocVec<T>(kmax); // sines of the Givens rotation metrices
	T* e1 = allocVec<T>(kmax + 1); // beta*e1 vec which will receive rotations
	//auto V = alloc_eigen<T, 2>({ kmax + 1,N });
	auto V = alloc_eigen<T, 2>({ N, kmax+1 });
	auto Ht = alloc_eigen<T, 2>({ kmax, kmax + 1 });
	//auto cs = alloc_eigen<T, 2>({ kmax, kmax + 1 });

	//calc_residual<T>(V[0], (const T**)A, (const T*)b, (const T*)x0, N); // r0 == b - A.dot(x0)
	V.col(0) = (b_eigen - A_eigen * x0_eigen);//.transpose();

	//double beta = norm<T>(V[0], N);
	double beta = V.col(0).norm();
	//mult_factor_to_row<T>(V, 1.0 / beta, 0, kmax + 1, N); // V[0] == r0/|r0|
	V.col(0) *= 1.0 / beta;
	e1[0] = static_cast<T>(beta);
	double resnorm = beta;
	double resnorm0 = beta;
	int k = 1;

	while (resnorm > tol* resnorm0&& k <= kmax) {

		auto Ht_km1 = Ht.row(k - 1);
		//auto Vk = V[k];

		// Arnoldi start. 
		// this is the time consuming part which has to be distributed and parallelized.
		//dense_mat_vec<T>(V[k], A, V[k - 1], N, N);
		V.col(k) = A_eigen * V.col(k - 1);

		if (algo == "cgs" || algo == "CGS") { // CGS
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				//Ht_km1[i - 1] = dense_vecconj_vec<T>(V[k], V[i - 1], N);
				Ht_km1(i - 1) = V.col(k).conjugate().dot(V.col(i - 1));
			}
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				//axpy<T>(V[k], -Ht_km1[i - 1], V[i - 1], V[k], N); // V[k] = -conj(H[i,k-1]) * V[i] + V[k]
				V.col(k) = -myconj(Ht_km1(i - 1)) * V.col(i - 1) + V.col(k);
			}
		}
		else if (algo == "mgs" || algo == "MGS") { // MGS
			for (int i = 1; i < k + 1; i++) {
				//Ht_km1[i - 1] = dense_vecconj_vec<T>(V[k], V[i - 1], N);
				//axpy<T>(V[k], -Ht_km1[i - 1], V[i - 1], V[k], N); // V[k] = -conj(H[i-1,k-1]) * V[i-1] + V[k]
				Ht_km1(i - 1) = V.col(k).conjugate().dot(V.col(i - 1));
				V.col(k) = -myconj(Ht_km1(i - 1)) * V.col(i - 1) + V.col(k);
			}
		}
		else {
			cout << "Specified algorithm is not implemented." << endl;
			break;
		}

		//double h_kp1k = norm<T>(Vk, N);
		double h_kp1k = V.col(k).norm();
		if (h_kp1k == 0) {
			break; // lucky breakdown
		}
		else {
			//Ht_km1[k] = h_kp1k;
			//mult_factor_to_row<T>(V, 1.0 / h_kp1k, k, kmax + 1, N);
			Ht_km1(k) = h_kp1k;
			V.col(k) *= 1.0 / h_kp1k;
		}
		// Arnoldi done.
		applyGivens_eigen<T>(Ht, cs, sn, e1, k);
		resnorm = abs(e1[k]);
		cout << "relResNorm after iter " << k << ": " << resnorm / resnorm0 << endl;
		k++;
	}

	// solve triangular
	//triangularSolve_inplace_lower<T>(Ht, e1, k - 1);

	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > e1_eigen(e1, kmax + 1, 1);
	e1_eigen.head(k - 1) = Ht.block(0, 0, k - 1, k - 1).colPivHouseholderQr().solve(e1_eigen.head(k - 1));


	// write resulting x to x0
	for (int i = 0; i < k - 1; i++) {
		//axpy(x0, myconj(e1[i]), V[i], x0, N);
		x0_eigen = myconj(e1_eigen(i)) * V.col(i) + x0_eigen;
	}

	//deleteMat<T>(V);// , kmax + 1);
	//deleteMat<T>(Ht);// , kmax);
	deleteVec<T>(cs);
	deleteVec<T>(sn);
	deleteVec<T>(e1);

	return k - 1;
}
#endif



# ifdef MPI_INCLUDED
// MPI version of gmres.
template <class T>
int gmres_mpi(
	const T** A,
	const T* b,
	T* x0/*initial guess. usually a zero vector. result will be stored here*/,
	int kmax/*max iteration*/,
	const double tol/*relative tolerance*/,
	const int N/*matrix size*/,
	const string& algo, /*choose from {cgs,mgs}*/
	const int rank,/*rank of this process*/
	const int nProc,/*number of processes*/
	const string& decompositionType, /*choose from {2r,2c,1r,1c}*/
	const int assumeSparse
) {

	if (kmax <= 0) {
		kmax = N;
	}

	int H = N;
	int W = N;

	function<tuple<int, int>(int)> fncDecomposition = getNumDomains_x_y_parallel_2D_row;
	if (decompositionType == "2r") {
		fncDecomposition = getNumDomains_x_y_parallel_2D_row;
	}
	else if (decompositionType == "2c") {
		fncDecomposition = getNumDomains_x_y_parallel_2D_col;
	}
	else if (decompositionType == "1r") {
		fncDecomposition = getNumDomains_x_y_parallel_1D_row;
	}
	else if (decompositionType == "1c") {
		fncDecomposition = getNumDomains_x_y_parallel_1D_col;
	}
	else {
		if (rank == 0) {
			cout << "  Specified decompositionType is not defined. Using default 2D-row decomposition." << endl;
		}
	}

	auto [nDomains_x, nDomains_y] = fncDecomposition(nProc);

	if (rank == 0) {
		cout << "  nDomains(x,y): " << nDomains_x << "," << nDomains_y << endl;
	}

	int domainIdx = rank;
	auto [domainIdx_x, domainIdx_y] = getDomainIndices_x_y_from_domainIdx(domainIdx, nDomains_x, nDomains_y);
	auto [rh, rt, ch, ct, h, w] = getDomainRange(H, W, domainIdx_x, domainIdx_y, nDomains_x, nDomains_y);

	auto A_domain = allocMat<T>(h, w);
	copyMat<T>(A_domain, A, 0, rh, 0, ch, h, w);


	T* spMat_csrVal = 0;
	int* spMat_csrColInd = 0;
	int* spMat_csrRowPtr = 0;
	int spMat_nnz = N * N;
	tuple<T*, int*, int*, int, int> spMat_domain;
	if (assumeSparse) {
		spMat_domain = denseToSparse(const_cast<T**>(A_domain), h, w);
		spMat_csrVal = get<0>(spMat_domain);
		spMat_csrColInd = get<1>(spMat_domain);
		spMat_csrRowPtr = get<2>(spMat_domain);
		spMat_nnz = get<4>(spMat_domain);
	}



	T** V  = allocMat<T>(kmax + 1, N); // Krylov bases
	
	//auto V_domain = allocMat<T>(kmax+1, w);
	auto AV_domain = allocMat<T>(kmax + 1, h);

	T** Ht = allocMat<T>(kmax, kmax + 1); // transpose of the Hessenberg matrix
	T* Ht_km1_buf = allocVec<T>(kmax + 1);
	T* Vk_norm_buf = allocVec<T>(1);
	T* cs  = allocVec<T>(kmax); // cosines of the Givens rotation metrices
	T* sn  = allocVec<T>(kmax); // sines of the Givens rotation metrices
	T* e1  = allocVec<T>(kmax + 1); // beta*e1 vec which will receive rotations

	calc_residual<T>(V[0], (const T * *)A, (const T*)b, (const T*)x0, N); // r0 == b - A.dot(x0)
	T beta = norm<T>(V[0], N);
	mult_factor_to_row<T>(V, 1.0 / beta, 0, kmax + 1, N); // V[0] == r0/|r0|

	e1[0] = static_cast<T>(beta);
	T resnorm = beta;
	T resnorm0 = beta;
	int k = 1;

	int color = rank / nDomains_y;
	const int key = rank % nDomains_y;
	MPI_Comm row_comm;/* Processes in same row */
	MPI_Comm col_comm;/* processes in same col */
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, key, rank, &col_comm);
	int row_rank, row_size, col_rank, col_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);
	MPI_Comm_rank(col_comm, &col_rank);
	MPI_Comm_size(col_comm, &col_size);

	MPI_Datatype mpiType;
	if constexpr (is_same_v<T, complex<double>>) {
		mpiType = MPI_DOUBLE_COMPLEX;
	}
	else if constexpr (is_same_v<T, complex<float>>) {
		mpiType = MPI_COMPLEX;
	}
	else if constexpr (is_same_v<T, double>) {
		mpiType = MPI_DOUBLE;
	}
	else if constexpr (is_same_v<T, float>) {
		mpiType = MPI_FLOAT;
	}

	//copyMat<T>(V_domain, (const T**)V, 0, 0, 0, ch, kmax+1, w);

	if (rank == 0 && H<=4) { // debug code
		printMat_to_stdOut_dense(A, H, W);
		printVec_to_stdOut_dense(V[0], H, "V[0]");
	}

	MPI_Status* status = new MPI_Status[nDomains_x];

	auto tol_T = static_cast<T>(tol);

	while (resnorm > tol_T * resnorm0 && k <= kmax) {

		auto Ht_km1 = Ht[k - 1];
		auto Vk = V[k];

		// Arnoldi start. 

		// dense mat vec product of domain(x,y) of A and domain(y) of V[k-1]
		if (!assumeSparse) {
			dense_mat_vec<T>(AV_domain[k], (const T**)A_domain, (const T*)(&(V[k - 1][ch])), h, w); // [3,11,19,27]
	//		dense_mat_vec<T>(AV_domain[k], (const T**)A_domain, (const T*) V_domain[k - 1], h, w);
		}
		else {
			sparse_mat_vec<T>(AV_domain[k], spMat_csrVal, spMat_csrColInd, spMat_csrRowPtr, &(V[k - 1][ch]), h);
		}

		// reduce AV_domain[k][0~h] from all ranks that are the same domainRow, to V[k][rh~rh+h] of rank0
		//MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0; i < nDomains_x; i++) {
			if (color == i) {
				//MPI_Reduce(&(AV_domain[k][0]), &(V[k][rh]), h, mpiType, MPI_SUM, 0, row_comm); // reduction, but only collecting block-row
				MPI_Allreduce(&(AV_domain[k][0]), &(V[k][rh]), h, mpiType, MPI_SUM, row_comm); // reduction, but only collecting block-row
			}
		}
		MPI_Barrier(row_comm);
		
		// at this point, the correct V[k][0~H-1] is shared by all processes.
		// can this be relaxed that each process has only their assigned V[k][rh~rh+h](?)

		if (algo == "cgs" || algo == "CGS") { // CGS
			MPI_Barrier(col_comm);
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				Ht_km1[i - 1] = dense_vecconj_vec<T>(&(V[k][rh]), &(V[i - 1][rh]), h); // this is the sum of one rowBlock. # [3,11,19,27].[1/2,1/2,1/2,1/2] = 7+23 = 30 
				MPI_Allreduce(&(Ht_km1[i - 1]), &(Ht_km1_buf[i - 1]), 1, mpiType, MPI_SUM, col_comm); // reduction, collecting Ht_km1[i-1] of all rowBlocks 
			}
			MPI_Barrier(col_comm);
			for (int i = 1; i < k + 1; i++) {
				Ht_km1[i - 1] = Ht_km1_buf[i - 1];
			}
			// at this point, the correct Ht_km1[i-1] is shared by all processes.
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				axpy<T>( &(V[k][rh]), -Ht_km1[i - 1], &(V[i - 1][rh]), &(V[k][rh]), h); // V1 = V1 - H *V0 = [-12,-4,4,12]
				//axpy<T>(&(V[k][0]), -Ht_km1[i - 1], &(V[i - 1][0]), &(V[k][0]), H); // V1 = V1 - H *V0 = [-12,-4,4,12]
			}
			// now, each rank has its own V[k][rh~rh+h] updated
			MPI_Barrier(MPI_COMM_WORLD);
			// sync V[k] of all column processes
			// write V[k][rh~] of each process to V[k][rh~] of rank0
			for (int idx_x = 1; idx_x < nDomains_x; idx_x++) { // send result of sum(Adomain * vdomain) to V[k] of rank0
				int tag = idx_x;
				auto [rh_of_domain_idx_x, o, o2, o3, h_of_domain_idx_x, o4] = getDomainRange(H, W, idx_x, 0, nDomains_x, nDomains_y);
				if (rank == 0) {
					MPI_Recv(&(V[k][rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, nDomains_y * idx_x, tag, MPI_COMM_WORLD, &status[tag]);
				}
				else if (rank == nDomains_y * idx_x) {
					MPI_Send(&(V[k][rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, 0, tag, MPI_COMM_WORLD);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&(V[k][0]), H, mpiType, 0, MPI_COMM_WORLD); // broadcast V[k][:] from rank0 to all
			MPI_Barrier(MPI_COMM_WORLD);
			

			// at this point, the correct V[k] is shared by all processes
			// maybe this can be relaxed.
		}
		else if (algo == "mgs" || algo == "MGS") { // MGS
			for (int i = 1; i < k + 1; i++) {
				Ht_km1[i - 1] = dense_vecconj_vec<T>(V[k], V[i - 1], N); // global reduction here
				axpy<T>(V[k], -Ht_km1[i - 1], V[i - 1], V[k], N);
			}
		}
		else {
			cout << "  Specified algorithm is not implemented." << endl;
			break;
		}

		// next steps is the computation of h_kp1k, the norm of V[k]
		// double h_kp1k = norm<T>(V[k], N); // global reduction here // h = sqrt(144+16+16+144) = sqrt(320) = 17.889
		// first, each rank computes the innerproduct of its block.
		auto h_kp1k_rowBlock_pow = dense_vecconj_vec<T>(&(V[k][rh]), &(V[k][rh]), h);
		MPI_Barrier(col_comm);
		MPI_Allreduce(&(h_kp1k_rowBlock_pow), &(Vk_norm_buf[0]), 1, MPI_DOUBLE, MPI_SUM, col_comm); // reduction, collecting h_kp1k_rowBlock of all rowBlocks 
		MPI_Barrier(col_comm);
		auto h_kp1k = sqrt(Vk_norm_buf[0]);
		// now, all ranks have correct h_kp1k.
		
		if (h_kp1k == 0) {
			break; // lucky breakdown
		}
		else {
			Ht_km1[k] = h_kp1k;
			mult_factor_to_row<T>(V, 1.0 / h_kp1k, k, kmax + 1, N); // normalize V[k].
		}
		// Arnoldi done.
		applyGivens<T>(Ht, cs, sn, e1, k);
		resnorm = abs(e1[k]);
		if (rank == 0) {
			//cout << "relResNorm after iter " << k << ": " << resnorm / resnorm0 << endl;
		}
		k++;
	}

	if (rank == 0) {
		cout << "  converged. relResNorm after iter " << k-1 << ": " << resnorm / resnorm0 << endl;
	}


	// solve triangular
	triangularSolve_inplace_lower<T>(Ht, e1, k - 1);

	// write resulting x to x0
	for (int i = 0; i < k - 1; i++) {
		axpy(x0, myconj(e1[i]), V[i], x0, N);
	}

	deleteMat<T>(V);// , kmax + 1);
	deleteMat<T>(Ht);// , kmax);
	deleteVec<T>(cs);
	deleteVec<T>(sn);
	deleteVec<T>(e1);

	return k - 1;
}
#endif




# ifdef MPI_INCLUDED
// MPI version of l1-GMRES.
template <class T>
int gmres_l1_mpi(
	const T** A,
	const T* b,
	T* x0/*initial guess. usually a zero vector. result will be stored here*/,
	int kmax/*max iteration*/,
	const double tol/*relative tolerance*/,
	const int N/*matrix size*/,
	const string& algo, /*choose from {cgs,mgs}*/
	const int rank,/*rank of this process*/
	const int nProc,/*number of processes*/
	const string& decompositionType, /*choose from {2r,2c,1r,1c}*/
	const int assumeSparse
) {

	if (kmax <= 0) {
		kmax = N;
	}

	int H = N;
	int W = N;

	function<tuple<int, int>(int)> fncDecomposition = getNumDomains_x_y_parallel_2D_row;
	if (decompositionType == "2r") {
		fncDecomposition = getNumDomains_x_y_parallel_2D_row;
	}
	else if (decompositionType == "2c") {
		fncDecomposition = getNumDomains_x_y_parallel_2D_col;
	}
	else if (decompositionType == "1r") {
		fncDecomposition = getNumDomains_x_y_parallel_1D_row;
	}
	else if (decompositionType == "1c") {
		fncDecomposition = getNumDomains_x_y_parallel_1D_col;
	}
	else {
		if (rank == 0) {
			cout << "  Specified decompositionType is not defined. Using default 2D-row decomposition." << endl;
		}
	}

	auto [nDomains_x, nDomains_y] = fncDecomposition(nProc);

	if (rank == 0) {
		cout << "  nDomains(x,y): " << nDomains_x << "," << nDomains_y << endl;
	}

	int domainIdx = rank;
	auto [domainIdx_x, domainIdx_y] = getDomainIndices_x_y_from_domainIdx(domainIdx, nDomains_x, nDomains_y);
	auto [rh, rt, ch, ct, h, w] = getDomainRange(H, W, domainIdx_x, domainIdx_y, nDomains_x, nDomains_y);

	auto A_domain = allocMat<T>(h, w);
	copyMat<T>(A_domain, A, 0, rh, 0, ch, h, w);


	T* spMat_csrVal = 0;
	int* spMat_csrColInd = 0;
	int* spMat_csrRowPtr = 0;
	int spMat_nnz = N * N;
	tuple<T*, int*, int*, int, int> spMat_domain;
	if (assumeSparse) {
		spMat_domain = denseToSparse(const_cast<T**>(A_domain), h, w);
		spMat_csrVal = get<0>(spMat_domain);
		spMat_csrColInd = get<1>(spMat_domain);
		spMat_csrRowPtr = get<2>(spMat_domain);
		spMat_nnz = get<4>(spMat_domain);
	}



	T** V = allocMat<T>(kmax + 1, N); // Krylov bases

	//auto V_domain = allocMat<T>(kmax+1, w);
	auto AV_domain = allocMat<T>(kmax + 1, h);

	T** Ht = allocMat<T>(kmax, kmax + 1); // transpose of the Hessenberg matrix
	T* Ht_km1_buf = allocVec<T>(kmax + 1);
	T* Vk_norm_buf = allocVec<T>(1);
	T* cs = allocVec<T>(kmax); // cosines of the Givens rotation metrices
	T* sn = allocVec<T>(kmax); // sines of the Givens rotation metrices
	T* e1 = allocVec<T>(kmax + 1); // beta*e1 vec which will receive rotations

	calc_residual<T>(V[0], (const T**)A, (const T*)b, (const T*)x0, N); // r0 == b - A.dot(x0)
	T beta = norm<T>(V[0], N);
	mult_factor_to_row<T>(V, 1.0 / beta, 0, kmax + 1, N); // V[0] == r0/|r0|

	e1[0] = static_cast<T>(beta);
	T resnorm = beta;
	T resnorm0 = beta;
	int k = 1;

	int color = rank / nDomains_y;
	const int key = rank % nDomains_y;
	MPI_Comm row_comm;/* Processes in same row */
	MPI_Comm col_comm;/* processes in same col */
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, key, rank, &col_comm);
	int row_rank, row_size, col_rank, col_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);
	MPI_Comm_rank(col_comm, &col_rank);
	MPI_Comm_size(col_comm, &col_size);

	MPI_Datatype mpiType;
	if constexpr (is_same_v<T, complex<double>>) {
		mpiType = MPI_DOUBLE_COMPLEX;
	}
	else if constexpr (is_same_v<T, complex<float>>) {
		mpiType = MPI_COMPLEX;
	}
	else if constexpr (is_same_v<T, double>) {
		mpiType = MPI_DOUBLE;
	}
	else if constexpr (is_same_v<T, float>) {
		mpiType = MPI_FLOAT;
	}

	//copyMat<T>(V_domain, (const T**)V, 0, 0, 0, ch, kmax+1, w);

	if (rank == 0 && H <= 4) { // debug code
		printMat_to_stdOut_dense(A, H, W);
		printVec_to_stdOut_dense(V[0], H, "V[0]");
	}

	MPI_Status* status = new MPI_Status[nDomains_x];

	auto tol_T = static_cast<T>(tol);

	while (resnorm > tol_T* resnorm0 && k <= kmax) {

		auto Ht_km1 = Ht[k - 1];
		auto Vk = V[k];

		// Arnoldi start. 

		// dense mat vec product of domain(x,y) of A and domain(y) of V[k-1]
		if (!assumeSparse) {
			dense_mat_vec<T>(AV_domain[k], (const T**)A_domain, (const T*)(&(V[k - 1][ch])), h, w); // [3,11,19,27]
	//		dense_mat_vec<T>(AV_domain[k], (const T**)A_domain, (const T*) V_domain[k - 1], h, w);
		}
		else {
			sparse_mat_vec<T>(AV_domain[k], spMat_csrVal, spMat_csrColInd, spMat_csrRowPtr, &(V[k - 1][ch]), h);
		}

		// at this point, each process has a part of AV, before summing up all column blocks.

		// reduce AV_domain[k][0~h] from all ranks that are the same domainRow, to V[k][rh~rh+h] of rank0
		//MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0; i < nDomains_x; i++) {
			if (color == i) {
				//MPI_Reduce(&(AV_domain[k][0]), &(V[k][rh]), h, mpiType, MPI_SUM, 0, row_comm); // reduction, but only collecting block-row
				MPI_Allreduce(&(AV_domain[k][0]), &(V[k][rh]), h, mpiType, MPI_SUM, row_comm); // reduction, but only collecting block-row
			}
		}
		MPI_Barrier(row_comm);

		// at this point, AV[k][rh~rh+h] is summed up and the correct V[k][rh~rh+h] is possessed by all processes.
		// this means that the correct V[k][0~H-1] is possessed by all processes.
		// can this be relaxed that each process has only their assigned V[k][rh~rh+h](?)

		MPI_Barrier(col_comm);
		// compute z2
		auto z2 = dense_vecconj_vec<T>(&(V[k][rh]), &(V[k][rh]), h);
		T z2buf = 0;
		MPI_Allreduce(&(z2), &(z2buf), 1, mpiType, MPI_SUM, col_comm); // reduction, collecting z2 of all rowBlocks 
		// compute Ht[k-1][:k]
		for (int i = 1; i < k + 1; i++) { // no dependency over i
			Ht_km1[i - 1] = dense_vecconj_vec<T>(&(V[k][rh]), &(V[i - 1][rh]), h); // this is the sum of one rowBlock. # [3,11,19,27].[1/2,1/2,1/2,1/2] = 7+23 = 30 
			MPI_Allreduce(&(Ht_km1[i - 1]), &(Ht_km1_buf[i - 1]), 1, mpiType, MPI_SUM, col_comm); // reduction, collecting Ht_km1[i-1] of all rowBlocks 
		}
		MPI_Barrier(col_comm);
		z2 = z2buf;
		for (int i = 1; i < k + 1; i++) {
			Ht_km1[i - 1] = Ht_km1_buf[i - 1];
		}
		// at this point, the correct Ht_km1[i-1] is shared by all processes.
		T h_kp1k = sqrt(abs(z2 - dense_vecconj_vec(Ht_km1, Ht_km1, k)));
		if (h_kp1k == 0) {
			break; // lucky breakdown
		}
		Ht_km1[k] = h_kp1k;
			
		for (int i = 1; i < k + 1; i++) { // no dependency over i
			axpy<T>(&(V[k][rh]), -Ht_km1[i - 1], &(V[i - 1][rh]), &(V[k][rh]), h); // V1 = V1 - H *V0 = [-12,-4,4,12]
		}

		// now, each rank has its own V[k][rh~rh+h] updated
			
		MPI_Barrier(MPI_COMM_WORLD);

		// we should be able to implement this better.
		// sync V[k] of all column processes
		// write V[k][rh~] of each process to V[k][rh~] of rank0
		for (int idx_x = 1; idx_x < nDomains_x; idx_x++) { // send result of sum(Adomain * vdomain) to V[k] of rank0
			int tag = idx_x;
			auto [rh_of_domain_idx_x, o, o2, o3, h_of_domain_idx_x, o4] = getDomainRange(H, W, idx_x, 0, nDomains_x, nDomains_y);
			if (rank == 0) {
				MPI_Recv(&(V[k][rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, nDomains_y * idx_x, tag, MPI_COMM_WORLD, &status[tag]);
			}
			else if (rank == nDomains_y * idx_x) {
				MPI_Send(&(V[k][rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, 0, tag, MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&(V[k][0]), H, mpiType, 0, MPI_COMM_WORLD); // broadcast V[k][:] from rank0 to all  // maybe this is redundant
			
		MPI_Barrier(MPI_COMM_WORLD);

		// at this point, the correct V[k] is shared by all processes, but this can be relaxed.

		mult_factor_to_row<T>(V, 1.0 / h_kp1k, k, kmax + 1, N); // normalize V[k].
		// should this be done only to a block of the (row) vector?
	
		// Arnoldi done.
		applyGivens<T>(Ht, cs, sn, e1, k);
		resnorm = abs(e1[k]);
		if (rank == 0) {
			//cout << "relResNorm after iter " << k << ": " << resnorm / resnorm0 << endl;
		}
		k++;
	}

	if (rank == 0) {
		cout << "  converged. relResNorm after iter " << k - 1 << ": " << resnorm / resnorm0 << endl;
	}


	// solve triangular
	triangularSolve_inplace_lower<T>(Ht, e1, k - 1);

	// write resulting x to x0
	for (int i = 0; i < k - 1; i++) {
		axpy(x0, myconj(e1[i]), V[i], x0, N);
	}

	deleteMat<T>(V);// , kmax + 1);
	deleteMat<T>(Ht);// , kmax);
	deleteVec<T>(cs);
	deleteVec<T>(sn);
	deleteVec<T>(e1);

	return k - 1;
}
#endif





#if defined(USE_EIGEN) && defined(MPI_INCLUDED)
// MPI version of gmres.
template <class T>
int gmres_mpi_eigen(
	T** A,
	T* b,
	T* x0/*initial guess. usually a zero vector. result will be stored here*/,
	int kmax/*max iteration*/,
	const double tol/*relative tolerance*/,
	const int N/*matrix size*/,
	const string& algo, /*choose from {cgs,mgs}*/
	const int rank,/*rank of this process*/
	const int nProc,/*number of processes*/
	const string& decompositionType, /*choose from {2r,2c,1r,1c}*/
	const int assumeSparse
) {

	if (kmax <= 0) {
		kmax = N;
	}

	int H = N;
	int W = N;


	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > A_eigen(A[0], N, N);
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > b_eigen(b, N, 1);
	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > x0_eigen(x0, N, 1);

	function<tuple<int, int>(int)> fncDecomposition = getNumDomains_x_y_parallel_2D_row;
	if (decompositionType == "2r") {
		fncDecomposition = getNumDomains_x_y_parallel_2D_row;
	}
	else if (decompositionType == "2c") {
		fncDecomposition = getNumDomains_x_y_parallel_2D_col;
	}
	else if (decompositionType == "1r") {
		fncDecomposition = getNumDomains_x_y_parallel_1D_row;
	}
	else if (decompositionType == "1c") {
		fncDecomposition = getNumDomains_x_y_parallel_1D_col;
	}
	else {
		if (rank == 0) {
			cout << "  Specified decompositionType is not defined. Using default 2D-row decomposition." << endl;
		}
	}

	auto [nDomains_x, nDomains_y] = fncDecomposition(nProc);

	if (rank == 0) {
		cout << "  nDomains(x,y): " << nDomains_x << "," << nDomains_y << endl;
	}

	int domainIdx = rank;
	auto [domainIdx_x, domainIdx_y] = getDomainIndices_x_y_from_domainIdx(domainIdx, nDomains_x, nDomains_y);
	auto [rh, rt, ch, ct, h, w] = getDomainRange(H, W, domainIdx_x, domainIdx_y, nDomains_x, nDomains_y);

	//auto A_domain = allocMat<T>(h, w);
	//copyMat<T>(A_domain, A, 0, rh, 0, ch, h, w);
	//Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor> > A_domain_eigen(A_domain[0], h, w);
	//auto A_domain_eigen_T = A_domain_eigen.transpose().eval();
	auto A_domain_eigen = A_eigen.block(rh, ch, h, w);

	//Eigen::Map<Eigen::VectorXcf> v_eigen(v, W);

	//T** V = allocMat<T>(kmax + 1, N); // Krylov bases
	//auto V_eigen = alloc_eigen<T>({ kmax + 1,N });

	auto V_eigen = alloc_eigen<T, 2>({ kmax + 1, N });
	//auto Ht = alloc_eigen<T, 2>({ kmax, kmax + 1 });

	//auto V_domain = allocMat<T>(kmax + 1, w);
	auto AV_domain = allocMat<T>(kmax + 1, h);
	//Eigen::Matrix<cf, Eigen::Dynamic, 1> Av_eigen(H);
	//const vector<int> shape = { static_cast<int>(kmax+1), static_cast<int>(h) };
	//auto AV_domain_eigen = alloc_eigen<T, 2>(shape);
	auto AV_domain_eigen = alloc_eigen<T, 2>({kmax+1,h});

	T** Ht = allocMat<T>(kmax, kmax + 1); // transpose of the Hessenberg matrix
	T* Ht_km1_buf = allocVec<T>(kmax + 1);
	//auto Ht_km1_buf = alloc_eigen<T, 2>({ kmax, kmax + 1 });
	T* Vk_norm_buf = allocVec<T>(1);
	T* cs = allocVec<T>(kmax); // cosines of the Givens rotation metrices
	T* sn = allocVec<T>(kmax); // sines of the Givens rotation metrices
	T* e1 = allocVec<T>(kmax + 1); // beta*e1 vec which will receive rotations

	//calc_residual<T>(V[0], (const T**)A, (const T*)b, (const T*)x0, N); // r0 == b - A.dot(x0)
	V_eigen.row(0) = (b_eigen - A_eigen * x0_eigen).transpose();
	//T beta = norm<T>(V[0], N);
	auto beta = V_eigen.row(0).norm();
	//mult_factor_to_row<T>(V, 1.0 / beta, 0, kmax + 1, N); // V[0] == r0/|r0|
	V_eigen.row(0) *= 1.0 / beta;
	//Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor> > V_eigen(V[0], kmax+1, N);

	e1[0] = static_cast<T>(beta);
	T resnorm = beta;
	T resnorm0 = beta;
	int k = 1;

	int color = rank / nDomains_y;
	const int key = rank % nDomains_y;
	MPI_Comm row_comm;/* Processes in same row */
	MPI_Comm col_comm;/* processes in same col */
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, key, rank, &col_comm);
	int row_rank, row_size, col_rank, col_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);
	MPI_Comm_rank(col_comm, &col_rank);
	MPI_Comm_size(col_comm, &col_size);

	MPI_Datatype mpiType;
	if constexpr (is_same_v<T, complex<double>>) {
		mpiType = MPI_DOUBLE_COMPLEX;
	}
	else if constexpr (is_same_v<T, complex<float>>) {
		mpiType = MPI_COMPLEX;
	}
	else if constexpr (is_same_v<T, double>) {
		mpiType = MPI_DOUBLE;
	}
	else if constexpr (is_same_v<T, float>) {
		mpiType = MPI_FLOAT;
	}

	//copyMat<T>(V_domain, (const T**)V, 0, 0, 0, ch, kmax+1, w);

	//if (rank == 0 && H <= 4) { // debug code
		//printMat_to_stdOut_dense(A, H, W);
		//printVec_to_stdOut_dense(V[0], H, "V[0]");
	//}

	MPI_Status* status = new MPI_Status[nDomains_x];

	//auto tol_T = static_cast<T>(tol);

	while (resnorm > tol * resnorm0 && k <= kmax) {
		//if (rank==0)
			//cout << "iter " << k << endl;
		auto Ht_km1 = Ht[k - 1];
		//auto Vk = V[k];

		// Arnoldi start. 

		// dense mat vec product of domain(x,y) of A and domain(y) of V[k-1]

		// set V[k-1] to V_eigen
		//for (int i = 0; i < w; i++) {
		//	V_eigen(k - 1, ch + i) = V[k - 1][ch + i];
		//}

		//dense_mat_vec<T>(AV_domain[k], (const T**)A_domain, (const T*)(&(V[k - 1][ch])), h, w); // AV_domain[k]: [h] // [3,11,19,27]
		AV_domain_eigen.row(k) = (A_domain_eigen * V_eigen.block(k-1,ch,1,w).transpose()).transpose(); // [h] = [h x w] . [w]
		//AV_domain_eigen.row(k) = V_eigen.block(k - 1, ch, 1, w) * A_domain_eigen_T; // [h] = [h x w] . [w]
		//AV_domain_eigen.eval();

//		dense_mat_vec<T>(AV_domain[k], (const T**)A_domain, (const T*) V_domain[k - 1], h, w);
		//dense_mat_vec_with_blocking<T>(AV_domain[k], A_domain, V_domain[k - 1], h, w);
		//dense_mat_vec_with_blocking_2<T>(AV_domain[k], A_domain, V_domain[k - 1], h, w);
		//dense_mat_vec_with_blocking_3<T>(AV_domain[k], A_domain, V_domain[k - 1], h, w);

		// reduce AV_domain[k][0~h] from all ranks that are the same domainRow, to V[k][rh~rh+h] of rank0
		//MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0; i < nDomains_x; i++) {
			if (color == i) {
				//MPI_Reduce(&(AV_domain[k][0]), &(V[k][rh]), h, mpiType, MPI_SUM, 0, row_comm); // reduction, but only collecting block-row
				//MPI_Allreduce(&(AV_domain[k][0]), &(V[k][rh]), h, mpiType, MPI_SUM, row_comm); // reduction, but only collecting block-row
//				MPI_Allreduce(&(AV_domain_eigen.data()[k*h]), &(V[k][rh]), h, mpiType, MPI_SUM, row_comm); // reduction, but only collecting block-row
				MPI_Allreduce(&(AV_domain_eigen.data()[k * h]), &(V_eigen.data()[k*N+rh]), h, mpiType, MPI_SUM, row_comm); // reduction, but only collecting block-row
			}
		}
		MPI_Barrier(row_comm);

		// at this point, the correct V[k][0~H-1] is shared by all processes.
		// can this be relaxed that each process has only their assigned V[k][rh~rh+h](?)

		if (algo == "cgs" || algo == "CGS") { // CGS
			MPI_Barrier(col_comm);
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				//Ht_km1[i - 1] = dense_vecconj_vec<T>(&(V[k][rh]), &(V[i - 1][rh]), h); // this is the sum of one rowBlock. # [3,11,19,27].[1/2,1/2,1/2,1/2] = 7+23 = 30 
				Ht_km1[i - 1] = (V_eigen.block(k, rh, 1, h).conjugate() * V_eigen.block(i - 1, rh, 1, h).transpose())(0, 0); // this is the sum of one rowBlock. # [3,11,19,27].[1/2,1/2,1/2,1/2] = 7+23 = 30 
				//Ht_km1[i - 1] = V_eigen.block(k, rh, 1, h).dot(V_eigen.block(i - 1, rh, 1, h));//.transpose())(0, 0); // this is the sum of one rowBlock. # [3,11,19,27].[1/2,1/2,1/2,1/2] = 7+23 = 30 
				MPI_Allreduce(&(Ht_km1[i - 1]), &(Ht_km1_buf[i - 1]), 1, mpiType, MPI_SUM, col_comm); // reduction, collecting Ht_km1[i-1] of all rowBlocks 
			}
			MPI_Barrier(col_comm);
			for (int i = 1; i < k + 1; i++) {
				Ht_km1[i - 1] = Ht_km1_buf[i - 1];
			}
			// at this point, the correct Ht_km1[i-1] is shared by all processes.
			for (int i = 1; i < k + 1; i++) { // no dependency over i
				//axpy<T>(&(V[k][rh]), -Ht_km1[i - 1], &(V[i - 1][rh]), &(V[k][rh]), h); // V1 = V1 - H *V0 = [-12,-4,4,12]
				V_eigen.block(k, rh, 1, h) = -Ht_km1[i - 1] * V_eigen.block(i - 1, rh, 1, h) + V_eigen.block(k, rh, 1, h);
				//axpy<T>(&(V[k][0]), -Ht_km1[i - 1], &(V[i - 1][0]), &(V[k][0]), H); // V1 = V1 - H *V0 = [-12,-4,4,12]
			}
			// now, each rank has its own V[k][rh~rh+h] updated
			MPI_Barrier(MPI_COMM_WORLD);
			// sync V[k] of all column processes
			// write V[k][rh~] of each process to V[k][rh~] of rank0
			for (int idx_x = 1; idx_x < nDomains_x; idx_x++) { // send result of sum(Adomain * vdomain) to V[k] of rank0
				int tag = idx_x;
				auto [rh_of_domain_idx_x, o, o2, o3, h_of_domain_idx_x, o4] = getDomainRange(H, W, idx_x, 0, nDomains_x, nDomains_y);
				if (rank == 0) {
					//MPI_Recv(&(V[k][rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, nDomains_y * idx_x, tag, MPI_COMM_WORLD, &status[tag]);
					MPI_Recv(&(V_eigen.data()[k*N+rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, nDomains_y * idx_x, tag, MPI_COMM_WORLD, &status[tag]);
				}
				else if (rank == nDomains_y * idx_x) {
					//MPI_Send(&(V[k][rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, 0, tag, MPI_COMM_WORLD);
					MPI_Send(&(V_eigen.data()[k*N+rh_of_domain_idx_x]), h_of_domain_idx_x, mpiType, 0, tag, MPI_COMM_WORLD);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			//MPI_Bcast(&(V[k][0]), H, mpiType, 0, MPI_COMM_WORLD); // broadcast V[k][:] from rank0 to all
			MPI_Bcast(&(V_eigen.data()[k*N]), H, mpiType, 0, MPI_COMM_WORLD); // broadcast V[k][:] from rank0 to all
			MPI_Barrier(MPI_COMM_WORLD);

			// at this point, the correct V[k] is shared by all processes
			// maybe this can be relaxed.
		}
		else if (algo == "mgs" || algo == "MGS") { // MGS
			for (int i = 1; i < k + 1; i++) {
				//Ht_km1[i - 1] = dense_vecconj_vec<T>(V[k], V[i - 1], N); // global reduction here
				Ht_km1[i - 1] = V_eigen.row(k).dot(V_eigen.row(i - 1));// global reduction here
				//axpy<T>(V[k], -Ht_km1[i - 1], V[i - 1], V[k], N);
				V_eigen.row(k) = -Ht_km1[i - 1] * V_eigen.row(i - 1) + V_eigen.row(k);
			}
		}
		else {
			cout << "  Specified algorithm is not implemented." << endl;
			break;
		}

		// next steps is the computation of h_kp1k, the norm of V[k]
		//double h_kp1k = norm<T>(V[k], N); // global reduction here // h = sqrt(144+16+16+144) = sqrt(320) = 17.889
		// first, each rank computes the innerproduct of its block.
		//auto h_kp1k_rowBlock_pow = dense_vecconj_vec<T>(&(V[k][rh]), &(V[k][rh]), h);
		auto h_kp1k_rowBlock_pow = (V_eigen.block(k, rh, 1, h).conjugate() * V_eigen.block(k, rh, 1, h).transpose())(0,0);
		//auto h_kp1k_rowBlock_pow = V_eigen.block(k, rh, 1, h).dot(V_eigen.block(k, rh, 1, h));// .transpose())(0, 0);
		MPI_Barrier(col_comm);
		MPI_Allreduce(&(h_kp1k_rowBlock_pow), &(Vk_norm_buf[0]), 1, MPI_DOUBLE, MPI_SUM, col_comm); // reduction, collecting h_kp1k_rowBlock of all rowBlocks 
		MPI_Barrier(col_comm);
		auto h_kp1k = sqrt(Vk_norm_buf[0]);
		// now, all ranks have correct h_kp1k.

		if (h_kp1k == 0) {
			//cout << "lucky breakdown!" << endl;
			break; // lucky breakdown
		}
		else {
			Ht_km1[k] = h_kp1k;
			//mult_factor_to_row<T>(V, 1.0 / h_kp1k, k, kmax + 1, N); // normalize V[k].
			V_eigen.row(k) *= 1.0 / h_kp1k;
		}
		// Arnoldi done.
		applyGivens<T>(Ht, cs, sn, e1, k);
		resnorm = abs(e1[k]);
		if (rank == 0) {
			//cout << "relResNorm after iter " << k << ": " << resnorm / resnorm0 << endl;
		}
		k++;
	}

	if (rank == 0) {
		cout << "  converged. relResNorm after iter " << k - 1 << ": " << resnorm / resnorm0 << endl;
	}


	// solve triangular
	triangularSolve_inplace_lower<T>(Ht, e1, k - 1);

	// write resulting x to x0
	for (int i = 0; i < k - 1; i++) {
		//axpy(x0, myconj(e1[i]), V[i], x0, N);
		x0_eigen = myconj(e1[i]) * V_eigen.row(i).transpose() + x0_eigen;
	}

	//deleteMat<T>(V);// , kmax + 1);
	deleteMat<T>(Ht);// , kmax);
	deleteVec<T>(cs);
	deleteVec<T>(sn);
	deleteVec<T>(e1);

	return k - 1;
}
#endif





// function for debugging
template<class T>
void writeMat_to_txtFile_dense(const string filePath, const T** mat, const int H, const int W) {
	ofstream outputfile(filePath);
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			outputfile << mat[i][j] << " ";
		}
		outputfile << "\n";
	}
	outputfile << "\n";
	outputfile.close();
}

// function for debugging
template<class T>
void writeMat_to_txtFile_sparse(const string filePath, const T** mat, const int H, const int W) {
	ofstream outputfile(filePath);
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			if(mat[i][j]==1)
				outputfile << i << " " << j << "\n";
		}
	}
	outputfile.close();
}

void writeTime_to_txtFile(const string filePath, double time) {
	ofstream outputfile(filePath);
	outputfile << time << "\n";
	outputfile.close();
}

vector<string> splitLine(string line, string delimiter) {
	vector<string> ret;
	string token;
	int pos = 0;
	while ((pos = (int)line.find(delimiter)) != std::string::npos) {
		token = line.substr(0, pos);
		ret.push_back(token);
		line.erase(0, pos + delimiter.length());
	}
	ret.push_back(line);
	return ret;
}

template<class T>
void setMat(T** mat, const int h, const int w, const T val) {
	for (int i = 0; i < h; i++) {
		T* row = mat[i];
		for (int j = 0; j < w; j++) {
			row[j] = val;
		}
	}
}

template<class T>
void setMat_2(T** mat, const int h, const int w, const T val, int rh, int ch) {
	for (int i = 0; i < h; i++) {
		T* row = mat[i+rh];
		for (int j = 0; j < w; j++) {
			row[j+ch] = val;
		}
	}
}

void print(string a) {
	cout << a << endl;
}

void print(string a, string b) {
	cout << a << " " << b << endl;
}

// function for debugging
template<class T>
void writeMatToStdOut(const T** mat, const int H, const int W, const int rh, const int ch) {
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			if (mat[i+rh][j+ch] == 1) {
				cout << (i+rh) << " " << (j+ch) << endl;
			}
		}
	}
}








// get the tag for MPI communication.
int getTag(int rank_recv, int rank2, int nDomain_x, int nDomain_y, int edgeIdx) {
	return rank_recv * 8 + edgeIdx;
}

template<class T>
tuple<T**, int, int> loadMatrix(string inFilePath) {
	ifstream fin(inFilePath);
	int H, W, nnz;
	while (fin.peek() == '%') {
		fin.ignore(2048, '\n');
	}
	fin >> H >> W >> nnz;
	T** mat = allocMat<T>(H, W);
	int k, l;
	T val;
	for (int i = 0; i < nnz; i++)
	{
		fin >> k >> l >> val;
		mat[k-1][l-1] = val;
	}
	fin.close();
	//cout << "matrix loaded. [H x W] = [" << H << " x " << W <<"]"<< endl;
	return { mat, H, W };
}

template<class T>
tuple<T*, int*, int*, int, int> loadSpMatrix(string inFilePath) {
	ifstream fin(inFilePath);
	int H, W, nnz;
	while (fin.peek() == '%') {
		fin.ignore(2048, '\n');
	}
	fin >> H >> W >> nnz;
	auto [csrVal, csrColInd, csrRowPtr] = allocSpMat<T>(H, nnz);

	int k, l;
	double val;
	for (int i = 0; i < H + 1; i++){
		csrRowPtr[i] = 0;
	}

	for (int i = 0; i < nnz; i++){
		fin >> k >> l >> val;
		csrVal[i] = val;
		csrColInd[i] = l-1;
		csrRowPtr[k]++;
	}

	for (int i = 1; i < H + 1; i++){
		csrRowPtr[i] += csrRowPtr[i-1];
	}

	fin.close();
	//cout << "matrix loaded. n =" << H << ", nnz = " << nnz <<" "<< endl;
	//printSpMat<T>(csrVal, csrColInd, csrRowPtr, H, nnz);
	return {csrVal, csrColInd, csrRowPtr, H, nnz};
}

template<class T>
tuple<T*, int> loadVector(string inFilePath, const int N) {
	if (inFilePath != "None") {
		ifstream fin(inFilePath);
		int H, W;
		while (fin.peek() == '%') {
			fin.ignore(2048, '\n');
		}
		fin >> H >> W;
		T* vec = allocVec<T>(H);
		double val;
		for (int i = 0; i < H; i++)
		{
			fin >> val;
			vec[i] = val;
			//cout << val << endl;
		}
		fin.close();
		//cout << "vector loaded. [H] = [" << H << "]" << endl;
		return { vec, H };
	}
	else {
		T* vec = allocVec<T>(N);
		for (int i = 0; i < N; i++)
		{
			vec[i] = 1.0;
		}
		return { vec, N };
	}
}



template<class T>
auto solve(const string& inFilePath_mat, const string& inFilePath_vec, int rank, int nProc, const string& decompositionType, const double tol, const int assumeSparse, string& algo, int maxIter) {

	//#define DEBUG
#ifdef DEBUG
	auto A = allocMat<T>(4, 4);
	int H = 4;
	int W = 4;
	auto b = allocVec<T>(4);
	int N = 4;
	int k = 0;
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			A[i][j] = k;
			k++;
		}
		b[i] = 1;
	}
#else
	auto [A, H, W] = loadMatrix<T>(inFilePath_mat);
	auto [b, N] = loadVector<T>(inFilePath_vec, W); // N == W
#endif


	if (rank == 0) {
		cout << "  Matrix size: " << H << " x " << W << endl;
	}

	auto x0 = allocVec<T>(N);
#ifdef MPI_INCLUDED
	auto tic = MPI_Wtime();
#else
	auto tic = omp_get_wtime();
#endif

#ifdef MPI_INCLUDED
	int nIter = 0;
	if (algo == "cgs") {
		nIter = gmres_mpi<T>((const T**)A, (const T*)b, x0, maxIter, tol, N, "cgs", rank, nProc, decompositionType, assumeSparse);
	}
#ifdef USE_EIGEN
	else if (algo=="cgs_eigen"){
		nIter = gmres_mpi_eigen<T>(A, b, x0, maxIter, tol, N, "cgs", rank, nProc, decompositionType, assumeSparse);
	}
#endif
	else if (algo == "cgs_l1" || algo=="l1") {
		nIter = gmres_l1_mpi<T>((const T**)(A), (const T*)(b), x0, maxIter, tol, N, "cgs", rank, nProc, decompositionType, assumeSparse);
	}
	else {
		assert(0);
	}
#else
	//int nIter = gmres<T>((const T * *)A, (const T*)b, x0, -1, tol, N, "mgs", assumeSparse);
	int nIter = 0;
	if (algo == "cgs") {
		nIter = gmres<T>((const T**)A, (const T*)b, x0, maxIter, tol, N, "cgs", assumeSparse);
	}
	else if (algo == "cgs_l1" || algo=="l1") {
		nIter = gmres<T>((const T**)A, (const T*)b, x0, maxIter, tol, N, "cgs_l1", assumeSparse);
	}
#ifdef USE_EIGEN
	else if (algo == "cgs_eigen") {
		nIter = gmres_eigen<T>(A, b, x0, maxIter, tol, N, "cgs", assumeSparse);
	}
	else if (algo == "cgs_eigen_ver2") {
		nIter = gmres_eigen_ver2<T>(A, b, x0, maxIter, tol, N, "cgs", assumeSparse);
	}
#endif
	else if (algo == "mgs") {
		nIter = gmres<T>((const T**)A, (const T*)b, x0, maxIter, tol, N, "mgs", assumeSparse);
	}
#ifdef USE_EIGEN
	else if (algo == "mgs_eigen") {
		nIter = gmres_eigen<T>(A, b, x0, maxIter, tol, N, "mgs", assumeSparse);
	}
	else if (algo == "mgs_eigen_ver2") {
		nIter = gmres_eigen_ver2<T>(A, b, x0, maxIter, tol, N, "mgs", assumeSparse);
	}
#endif
	else {
		assert(0);
	}
#endif
#ifdef MPI_INCLUDED
	auto toc = MPI_Wtime();
#else
	auto toc = omp_get_wtime();
#endif
	deleteMat<T>(A);// , H);
	deleteVec<T>(b);
	deleteVec<T>(x0);
	return toc - tic;
}

int main(int argc, char* argv[])
{
	int rank = 0;
	int size = 1;

#ifdef MPI_INCLUDED
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//cout << "rank, size: "<< rank << ", " << size << endl;
	if (rank == 0) {
		cout << "GMRES solver." << endl;
		cout << "  nProcesses: " << size << endl;
	}
#endif

	string decompositionType;
	//int decompositionDim = 2;
	int nDomains = 256;
	if (argc > 1) {
		//decompositionDim = stoi(argv[5]);
		int nThreads = stoi(argv[1]);
		if (rank == 0) {
			cout << "  nThreads: " << argv[1] << endl;
		}
		omp_set_num_threads(nThreads);
#ifdef USE_EIGEN
		Eigen::setNbThreads(nThreads);
#endif
	}
	if (argc > 2) {
		decompositionType = argv[2];
		if (rank == 0) {
			cout << "  decompositionType: " << decompositionType << endl;
		}
	}
	
	string inFilePath_mat;
	string inFilePath_vec;

//	if (argc > 2) {
		//cout << argv[1] << endl;
	//	inFilePath_mat = string(argv[1]);
	//	inFilePath_vec = string(argv[2]);
	//}
	//else {
	//}
	auto matName = string("sherman4");
	inFilePath_mat = string("matrices/sherman/sherman4.mtx");
	inFilePath_vec = string("matrices/sherman/sherman4_rhs1.mtx");
	if (argc > 3) {
		matName = argv[3];
		if (matName == "sherman1" || matName=="sherman2" || matName == "sherman3" || matName == "sherman4" || matName == "sherman5") {
			inFilePath_mat = string("matrices/sherman/") + matName  +string(".mtx");
			inFilePath_vec = string("matrices/sherman/") + matName + string("_rhs1.mtx");
		}
		else if (matName == "bcsstk18") {
			inFilePath_mat = string("matrices/bcsstk/bcsstk18.mtx");
			inFilePath_vec = string("None");
		}
	}
	if (rank==0)
		cout << "  matrixName: " << matName << endl;

	double tol = 1e-10;
	if (argc > 4) {
		tol = stof(argv[4]);
	}
	if (rank == 0)
		cout << "  tol: " << argv[4] << endl;

	int assumeSparse = 0;
	if (argc > 5) {
		if (string(argv[5])=="sparse") {
			assumeSparse = 1;
		}
	}
	if (rank == 0)
		cout << "  dense_or_sparse: " << string(argv[5]) << endl;

	string algo = "cgs";
	if (argc > 6) {
		algo = string(argv[6]);
	}
	if (rank == 0)
		cout << "  algo: " << algo << endl;

	int maxIter = -1;
	if (argc > 7) {
		maxIter = stoi(argv[7]);
	}

	//int time = 0;
	//auto time = solve<float>(inFilePath_mat, inFilePath_vec, rank, size, decompositionType, tol);
	auto time = solve<double>(inFilePath_mat, inFilePath_vec, rank, size, decompositionType, tol, assumeSparse, algo, maxIter);
	//auto time = solve<complex<double>>(inFilePath_mat, inFilePath_vec, rank, size);

	if (rank == 0) {
		cout << "  time: " << time << endl;
		//writeTime_to_txtFile("time_nProc=" + to_string(size) + "_decompDim=" + to_string(decompositionDim) + ".txt", time);
	}

#ifdef MPI_INCLUDED
	MPI_Finalize();
#endif
	//getchar();
	return 0;
}