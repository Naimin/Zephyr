#ifndef ZEPHYR_GPU_QUADRIC_H
#define ZEPHYR_GPU_QUADRIC_H

#include "../stdfx.h"
#include <GeometryMath.h>
#include "OpenMesh_GPU.h"
#include <tbb/parallel_for.h>

namespace Zephyr
{
	namespace GPU
	{
		struct Quadric_GPU
		{
			double mat[10];

			__device__
			Quadric_GPU(const double a=0, const double b=0, const double c=0, const double d=0)
				: mat{a*a, a*b, a*c, a*d, b*b, b*c, b*d, c*c, c*d, d*d }{}

			__device__
			Quadric_GPU operator+=(const Quadric_GPU q)
			{
				mat[0] += q.mat[0];
				mat[1] += q.mat[1];
				mat[2] += q.mat[2];
				mat[3] += q.mat[3];
				mat[4] += q.mat[4];
				mat[5] += q.mat[5];
				mat[6] += q.mat[6];
				mat[7] += q.mat[7];
				mat[8] += q.mat[8];
				mat[9] += q.mat[9];
				return *this;
			}

			__device__
			Quadric_GPU operator*=(const double s)
			{
				mat[0] *= s;
				mat[1] *= s;
				mat[2] *= s;
				mat[3] *= s;
				mat[4] *= s;
				mat[5] *= s;
				mat[6] *= s;
				mat[7] *= s;
				mat[8] *= s;
				mat[9] *= s;
				return *this;
			}

			__device__
			double evalute(Common::Vector3f v) const
			{
				double x(v[0]), y(v[1]), z(v[2]);
				return mat[0]*x*x + 2.0*mat[1]*x*y + 2.0*mat[2]*x*z + 2.0*mat[3]*x
								  +     mat[4]*y*y + 2.0*mat[5]*y*z + 2.0*mat[6]*y
					   							   +     mat[7]*z*z + 2.0*mat[8]*z
																    +     mat[9];
			}
		};

		struct QEM_Data_Package
		{
			QEM_Data_Package(size_t numQEM_Data) 
			{
				size_t QEM_size = numQEM_Data * sizeof(QEM_Data);
				cudaMalloc((void**)&mp_QEM_Data_GPU, QEM_size);
			}

			QEM_Data_Package(std::vector<QEM_Data>& QEM_Datas)
			{
				size_t QEM_size = QEM_Datas.size() * sizeof(QEM_Data);
				cudaMalloc((void**)&mp_QEM_Data_GPU, QEM_size);
				setup(QEM_Datas);
			}

			void setup(std::vector<QEM_Data>& QEM_Datas)
			{
				size_t QEM_size = QEM_Datas.size() * sizeof(QEM_Data);
				cudaMemcpy(mp_QEM_Data_GPU, &QEM_Datas[0], QEM_size, cudaMemcpyHostToDevice);
			}

			~QEM_Data_Package()
			{
				//cudaFree(mp_QEM_Data_GPU);
			}

			// device ptr
			QEM_Data* mp_QEM_Data_GPU;
		};

		
	}
}

#endif