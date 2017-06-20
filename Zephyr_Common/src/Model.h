#ifndef MODEL_H
#define MODEL_H

#include "Material.h"
#include "Mesh.h"
#include <vector>

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API Model
		{
		public:
			Model();
			virtual ~Model();

			// Mesh Accessor
			void addMesh(Mesh& mesh);
			Mesh& getMesh(const int i);
			const Mesh& getMesh(const int i) const;

			std::vector<Mesh>& getMeshes();
			const std::vector<Mesh>& getMeshes() const;
			int getMeshesCount() const;
			// Material Accessor
			void addMaterial(Material& material);
			Material& getMaterial(const int i);
			const Material& getMaterial(const int i) const;
			
			std::vector<Material>& getMaterials();
			const std::vector<Material>& getMaterials() const;
			int getMaterialsCount() const;

			// Texture Accessor
			void addTexture(const boost::filesystem::path& texturePath);
			Texture& getTexture(const int i);
			const Texture& getTexture(const int i) const;

			std::vector<Texture>& getTextures();
			const std::vector<Texture>& getTextures() const;
			int getTexturesCount() const;

		private:
			std::vector<Mesh> mMeshes;
			std::vector<Material> mMaterials;
			std::vector<Texture> mTextures;
		};
	}
}

#endif
