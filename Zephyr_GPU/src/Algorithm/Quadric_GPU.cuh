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

		struct QEM_Data_GPU
		{
			QEM_Data_GPU() {}

			QEM_Data_GPU(QEM_Data& QEM_data) : vertexCount((INDEX_TYPE)QEM_data.vertices.size()), indexCount((INDEX_TYPE)QEM_data.indices.size())
			{
				size_t vertexSize = QEM_data.vertices.size() * sizeof(QEM_data.vertices[0]);
				cudaMalloc((void**)&vertices, vertexSize);
				cudaMemcpy(vertices, &QEM_data.vertices[0], vertexSize, cudaMemcpyHostToDevice);
				size_t indexSize = QEM_data.indices.size() * sizeof(QEM_data.indices[0]);
				cudaMalloc((void**)&indices, indexSize);
				cudaMemcpy(indices, &QEM_data.indices[0], indexSize, cudaMemcpyHostToDevice);
			}

			void free()
			{
				cudaFree(vertices);
				cudaFree(indices);
			}

			Common::Vector3f* vertices;
			INDEX_TYPE vertexCount;
			INDEX_TYPE* indices;
			INDEX_TYPE indexCount;
			INDEX_TYPE vertexToKeepId;
		};

		struct QEM_Data_Package
		{
			QEM_Data_Package(std::vector<QEM_Data>& QEM_Datas)
			{
				m_QEM_Data_GPUs.resize(QEM_Datas.size());
				tbb::parallel_for((size_t)0, QEM_Datas.size(), [&](const size_t id)
				{
					m_QEM_Data_GPUs[id] = QEM_Data_GPU(QEM_Datas[id]);
				});
				QEM_Datas.clear();

				size_t QEM_size = m_QEM_Data_GPUs.size() * sizeof(m_QEM_Data_GPUs[0]);
				cudaMalloc((void**)&mp_QEM_Data_GPU, QEM_size);
				cudaMemcpy(mp_QEM_Data_GPU, &m_QEM_Data_GPUs[0], QEM_size, cudaMemcpyHostToDevice);
			}

			~QEM_Data_Package()
			{
				cudaFree(mp_QEM_Data_GPU);
				tbb::parallel_for((size_t)0, m_QEM_Data_GPUs.size(), [&](const size_t id)
				{
					m_QEM_Data_GPUs[id].free();
				});
			}

			// device ptr
			QEM_Data_GPU* mp_QEM_Data_GPU;
			std::vector<QEM_Data_GPU> m_QEM_Data_GPUs;
		};

		
	}
}

#endif