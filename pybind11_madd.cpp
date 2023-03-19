/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_madd(float* A, float* B, float* C, int M, int N, float coef1, float coef2);
// extern void cu_msub
// extern void cu_mcoef
extern void cu_mtile(float* A, float* C, int M, int N);
extern void cu_mabs(float* A, float* C, int M, int N);
extern void cu_msum(float* A, float* C, int M, int N);
extern void cu_mmul(float* A, float* B, float* C, int M, int N);
extern void cu_mdiv(float* A, float* B, float* C, int M, int N);
extern void cu_mtrans(float* A, float* C, int M, int N);
extern void cu_maddCons(float* A, float* C, int M, int N, float constant);

namespace py = pybind11;


py::array_t<float> madd_wrapper(py::array_t<float> a1, py::array_t<float> a2, float coef1, float coef2)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.size != buf2.size)
		throw std::runtime_error("Input shapes must match");


	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();



	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;

	cu_madd(A, B, C, M, N, coef1, coef2);

	return result;
}

py::array_t<float> msub_wrapper(py::array_t<float> a1, py::array_t<float> a2)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.size != buf2.size)
		throw std::runtime_error("Input shapes must match");

	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;

	cu_madd(A, B, C, M, N, 1, -1);

	return result;
}

py::array_t<float> mcoef_wrapper(py::array_t<float> a1, float coef)
{
	auto buf1 = a1.request();
	auto buf2 = a1.request();

	if (a1.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.size != buf2.size)
		throw std::runtime_error("Input shapes must match");

	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf1.ptr;
	float* C = (float*)buf3.ptr;

	cu_madd(A, B, C, M, N, coef, 0);

	return result;
}

py::array_t<float> mtile_wrapper(py::array_t<float> a1, int tiles)
{
	auto buf1 = a1.request();


	// NxM matrix
	int N = a1.shape()[0];
	int M = tiles;

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		2,          /* How many dimensions? */
		{ M, N },  /* Number of elements for each dimension */
		{ sizeof(float) * N, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* C = (float*)buf3.ptr;

	cu_mtile(A, C, M, N);

	return result;
}

py::array_t<float> mabs_wrapper(py::array_t<float> a1)
{
	auto buf1 = a1.request();

	if (a1.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* C = (float*)buf3.ptr;

	cu_mabs(A, C, M, N);

	return result;
}

py::array_t<float> msum_wrapper(py::array_t<float> a1)
{

	auto buf1 = a1.request();
	if (a1.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");


	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,         /* How many dimensions? */
		{ 1, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* C = (float*)buf3.ptr;

	cu_msum(A, C, M, N);

	return result;

}

py::array_t<float> mmul_wrapper(py::array_t<float> a1, py::array_t<float> a2)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.size != buf2.size)
		throw std::runtime_error("Input shapes must match");


	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;


	cu_mmul(A, B, C, M, N);

	return result;
}

py::array_t<float> mdiv_wrapper(py::array_t<float> a1, py::array_t<float> a2)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	if (buf1.size != buf2.size)
		throw std::runtime_error("Input shapes must match");


	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;


	cu_mdiv(A, B, C, M, N);

	return result;
}

py::array_t<float> mtrans_wrapper(py::array_t<float> a1)
{
	auto buf1 = a1.request();

	if (a1.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");



	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ M, N },  /* Number of elements for each dimension */
		{ sizeof(float) * N, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();


	float* A = (float*)buf1.ptr;
	float* C = (float*)buf3.ptr;

	cu_mtrans(A, C, M, N);

	return result;
}

py::array_t<float> maddCons_wrapper(py::array_t<float> a1, float constant)
{
	auto buf1 = a1.request();

	if (a1.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");



	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(float) * M, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();



	float* A = (float*)buf1.ptr;
	float* C = (float*)buf3.ptr;

	cu_maddCons(A, C, M, N, constant);

	return result;
}




PYBIND11_MODULE(cu_matrix_add, m) {
	m.def("madd", &madd_wrapper, "Add two NumPy arrays with coefficients");
	m.def("msub", &msub_wrapper, "Subtract two Numpy arrays");
	m.def("mcoef", &mcoef_wrapper, "Multiply matrix by coef");
	m.def("mtile", &mtile_wrapper, "Tile a vector into a matrix");
	m.def("mabs", &mabs_wrapper, "Take absolute value of matrix");
	m.def("msum", &msum_wrapper, "Sum Matrix across rows");
	m.def("mmul", &mmul_wrapper, "Multiply Matrices by element");
	m.def("mdiv", &mdiv_wrapper, "Divide Matrices by element");
	m.def("mtrans", &mtrans_wrapper, "Divide Matrices by element");
	m.def("maddCons", &maddCons_wrapper, "Add a constant to every element of matrix");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}
