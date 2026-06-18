// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/scene/rendering/SceneResources.hpp"

#define STB_IMAGE_IMPLEMENTATION

#include <GL/glew.h>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <stb_image.h>

#include <ace_loggers/ace_loggers.hpp>
#include <assimp/Importer.hpp>
#include <filesystem>
#include <iostream>

#include "ace_monitor/scene/rendering/GLShader.hpp"
#include "ace_monitor/scene/rendering/GLTexture.hpp"
#include "ace_monitor/scene/rendering/Material.hpp"
#include "ace_monitor/scene/rendering/Mesh3D.hpp"
#include "ace_monitor/scene/rendering/Model3D.hpp"
#include "ace_monitor/utils/TextureHelpers.hpp"

namespace ace_monitor {

class Model3DLoader {
 public:
  // Model loader code is based on: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/includes/learnopengl/model.h
  std::vector<Mesh3D::SharedPtr> meshes;
  std::string directory;

  bool Load(const std::string path) {
    // read file via ASSIMP
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
      path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
    // check for errors
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)  // if is Not Zero
    {
      LOG(WARNING) << "Failed to load model:" << path << "\nError:" << importer.GetErrorString();
      return false;
    }
    // retrieve the directory path of the filepath
    directory = path.substr(0, path.find_last_of('/'));

    // process ASSIMP's root node recursively
    ProcessNode(scene->mRootNode, scene);
    return true;
  }

  void ProcessNode(aiNode* node, const aiScene* scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
      // the node object only contains indices to index the actual objects in the scene.
      // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
      aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
      meshes.push_back(ProcessMesh(mesh, scene));
    }
    // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
      ProcessNode(node->mChildren[i], scene);
    }
  }

  Mesh3D::SharedPtr ProcessMesh(aiMesh* mesh_node, const aiScene* scene) const {
    // data to fill
    auto mesh3d = std::make_shared<Mesh3D>(false, false);
    auto material = std::make_shared<Material>();
    mesh3d->SetMaterial(material);

    auto& vertices = mesh3d->GetVerticies();
    auto& indices = mesh3d->GetIndicies();
    vertices.resize(mesh_node->mNumVertices);

    // walk through each of the mesh's vertices
    for (unsigned int i = 0; i < mesh_node->mNumVertices; i++) {
      Vertex3D& vertex = vertices[i];
      // positions
      vertex.position.x = mesh_node->mVertices[i].x;
      vertex.position.y = mesh_node->mVertices[i].y;
      vertex.position.z = mesh_node->mVertices[i].z;
      // normals
      if (mesh_node->HasNormals()) {
        vertex.normal.x = mesh_node->mNormals[i].x;
        vertex.normal.y = mesh_node->mNormals[i].y;
        vertex.normal.z = mesh_node->mNormals[i].z;
      }
      if (mesh_node->HasVertexColors(0)) {
        vertex.color.r = mesh_node->mColors[0][i].r;
        vertex.color.g = mesh_node->mColors[0][i].g;
        vertex.color.b = mesh_node->mColors[0][i].b;
      }
      if (mesh_node->mTextureCoords[0]) {
        vertex.uv0.x = mesh_node->mTextureCoords[0][i].x;
        vertex.uv0.y = mesh_node->mTextureCoords[0][i].y;
      } else {
        vertex.uv0 = glm::vec2(0.0F, 0.0F);
      }
    }
    for (unsigned int i = 0; i < mesh_node->mNumFaces; i++) {
      aiFace face = mesh_node->mFaces[i];
      // retrieve all indices of the face and store them in the indices vector
      for (unsigned int j = 0; j < face.mNumIndices; j++) {
        indices.push_back(face.mIndices[j]);
      }
    }
    // process materials
    aiMaterial* material_node = scene->mMaterials[mesh_node->mMaterialIndex];

    // load diffuse map only
    material->SetTextures(LoadMaterialTextures(material_node, aiTextureType_DIFFUSE));

    mesh3d->UpdateVerticies();
    mesh3d->UpdateIndicies();
    return mesh3d;
  }

  std::vector<GLTexture::SharedPtr> LoadMaterialTextures(aiMaterial* material_node, aiTextureType type) const {
    std::vector<GLTexture::SharedPtr> textures;
    for (unsigned int i = 0; i < material_node->GetTextureCount(type); i++) {
      aiString str;
      material_node->GetTexture(type, i, &str);
      // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
      auto texture = SceneResources::GetInstance().GetOrCreateTexture(this->directory + std::string("/") + str.C_Str());
      if (texture) {
        textures.push_back(texture);
      }
    }
    return textures;
  }
};
///
SceneResources& SceneResources::GetInstance() {
  static SceneResources instance;
  return instance;
}

std::shared_ptr<GLTexture> SceneResources::GetTexture(const std::string& name) const {
  auto it = textures_.find(name);
  if (it != textures_.end()) {
    return it->second;
  }
  return nullptr;
}
std::shared_ptr<GLShader> SceneResources::GetShader(const std::string& name) const {
  auto it = shaders_.find(name);
  if (it != shaders_.end()) {
    return it->second;
  }
  return nullptr;
}
std::shared_ptr<Model3D> SceneResources::GetModel(const std::string& name) const {
  auto it = models_.find(name);
  if (it != models_.end()) {
    return it->second;
  }
  return nullptr;
}

void SceneResources::AddTexture(std::shared_ptr<GLTexture> texture) { textures_[texture->GetName()] = texture; }
void SceneResources::AddShader(std::shared_ptr<GLShader> shader) { shaders_[shader->GetName()] = shader; }
void SceneResources::AddModel(std::shared_ptr<Model3D> model) { models_[model->GetName()] = model; }

std::shared_ptr<GLTexture> SceneResources::GetOrCreateTexture(const std::string& path) {
  auto texture = GetTexture(path);
  if (texture != nullptr) {
    return texture;
  }

  int width;
  int height;
  int components;
  unsigned char* data = stbi_load(path.c_str(), &width, &height, &components, 0);
  if (!data) {
    LOG(WARNING) << "Failed to load texture: " << path;
    return nullptr;
  }

  texture = std::make_shared<GLTexture>(path);
  auto texture_id = TextureHelpers::CreateTexture(width, height, components, data);

  glBindTexture(GL_TEXTURE_2D, texture_id);
  glGenerateMipmap(GL_TEXTURE_2D);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  stbi_image_free(data);

  texture->SetTextureID(texture_id);
  AddTexture(texture);
  LOG(INFO) << "Texture Loaded: " << path;
  return texture;
}
std::shared_ptr<Model3D> SceneResources::GetOrCreateModel(const std::string& path) {
  auto model = GetModel(path);
  if (model != nullptr) {
    return model;
  }
  Model3DLoader loader;
  if (!loader.Load(path)) {
    return nullptr;
  }

  model = std::make_shared<Model3D>(path);
  model->SetMeshes(loader.meshes);
  AddModel(model);
  LOG(INFO) << "Model Loaded: " << path;
  return model;
}

std::shared_ptr<GLShader> SceneResources::GetOrCreateShader(const std::string& path) {
  auto shader = GetShader(path);
  if (shader != nullptr) {
    return shader;
  }
  shader = std::make_shared<GLShader>(path);
  if (std::filesystem::exists(path + "/shader.vs")) {
    (void)shader->LoadShaderFromFile(GLShader::ShaderType::kVertex, path + "/shader.vs");
  }
  if (std::filesystem::exists(path + "/shader.fs")) {
    (void)shader->LoadShaderFromFile(GLShader::ShaderType::kFragment, path + "/shader.fs");
  }
  if (std::filesystem::exists(path + "/shader.gs")) {
    (void)shader->LoadShaderFromFile(GLShader::ShaderType::kGeometry, path + "/shader.gs");
  }
  if (!shader->LinkShaders()) {
    LOG(WARNING) << "Failed to link shader: " << path;
    return nullptr;
  }
  LOG(INFO) << "Shader Loaded: " << path;
  AddShader(shader);
  return shader;
}
}  // namespace ace_monitor
