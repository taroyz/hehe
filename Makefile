paris: paris.cpp
	mpic++ paris.cpp -std=c++17 -O3 -fopenmp -I $(EIGEN_INCDIR) -I$(CUDA_INCDIR) -L$(CUDA_LIB64DIR) -Wl,-rpath,$(CUDA_LIB64DIR)
all: paris
