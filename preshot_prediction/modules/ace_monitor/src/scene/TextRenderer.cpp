// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "ace_monitor/scene/TextRenderer.hpp"

#include <freetype2/ft2build.h>

#include "ace_loggers/ace_loggers.hpp"
#include "ace_monitor/scene/CameraNode.hpp"
#include "ace_monitor/scene/rendering/GLShader.hpp"
#include "ace_monitor/scene/rendering/SceneResources.hpp"
#include "ace_monitor/scene/rendering/ShaderUniforms.hpp"
#include FT_FREETYPE_H

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <iostream>
#include <map>

#include "ace_monitor/AceMonitor.hpp"

namespace ace_monitor {

class TextRenderer::TextRendererImpl {
 public:
  // Code derived from this source:
  // https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/7.in_practice/2.text_rendering/text_rendering.cpp
  class Character {
   public:
    ~Character() { glDeleteTextures(1, &texture_id); }
    unsigned int texture_id{0};  // ID handle of the glyph texture
    glm::vec2 size;              // size of glyph
    glm::vec2 bearing;           // Offset from baseline to left/top of glyph
    int advance;                 // Horizontal offset to advance to next glyph
  };

  std::unordered_map<unsigned char, std::unique_ptr<Character>> characters;
  unsigned int vao, vbo;

  GLShader::SharedPtr shader;
  const GLShader::UniformInfo* projection_uniform{nullptr};
  const GLShader::UniformInfo* color_uniform{nullptr};
};

TextRenderer::TextRenderer() { impl_ = std::make_unique<TextRendererImpl>(); }
TextRenderer::~TextRenderer() = default;

bool TextRenderer::Initialize(const std::string& font_path, int pixel_size) {
  FT_Library ft;
  FT_Face face;
  if (FT_Init_FreeType(&ft)) {
    LOG(WARNING) << "ERROR::FREETYPE: Could not init FreeType Library";
    return false;
  }
  if (FT_New_Face(ft, font_path.c_str(), 0, &face)) {
    LOG(WARNING) << "ERROR::FREETYPE: Failed to load font";
    return false;
  }  // set size to load glyphs as
  FT_Set_Pixel_Sizes(face, 0, pixel_size);

  // disable byte-alignment restriction
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // load first 128 characters of ASCII set
  for (unsigned char c = 0; c < 128; c++) {
    // Load character glyph
    if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
      LOG(WARNING) << "ERROR::FREETYTPE: Failed to load Glyph";
      continue;
    }
    // generate texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, static_cast<GLsizei>(face->glyph->bitmap.width),
                 static_cast<GLsizei>(face->glyph->bitmap.rows), 0, GL_RED, GL_UNSIGNED_BYTE,
                 face->glyph->bitmap.buffer);
    // set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // now store character for later use
    auto character = std::make_unique<TextRendererImpl::Character>();
    character->texture_id = texture;
    character->size = glm::vec2(face->glyph->bitmap.width, face->glyph->bitmap.rows);
    character->bearing = glm::vec2(face->glyph->bitmap_left, face->glyph->bitmap_top);
    character->advance = static_cast<int>(face->glyph->advance.x);
    impl_->characters[c] = std::move(character);
  }
  glBindTexture(GL_TEXTURE_2D, 0);

  // destroy FreeType once we're finished
  FT_Done_Face(face);
  FT_Done_FreeType(ft);

  glGenVertexArrays(1, &impl_->vao);
  glGenBuffers(1, &impl_->vbo);
  glBindVertexArray(impl_->vao);
  glBindBuffer(GL_ARRAY_BUFFER, impl_->vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  auto vs_path = ACEMonitor::GetInstance().GetDataPath() + "shaders/font/vertex.vs";
  auto fs_path = ACEMonitor::GetInstance().GetDataPath() + "shaders/font/fragment.fs";
  impl_->shader =
    SceneResources::GetInstance().GetOrCreateShader(ACEMonitor::GetInstance().GetDataPath() + "shaders/font");
  const auto* texture_uniform = impl_->shader->GetUniform("texture_diffuse");
  if (texture_uniform) {
    impl_->shader->Bind();
    ShaderUniforms::SetInt(texture_uniform->id, 0);
    impl_->shader->Unbind();
  }
  impl_->projection_uniform = impl_->shader->GetUniform("projection");
  impl_->color_uniform = impl_->shader->GetUniform("tint_color");
  return true;
}

void TextRenderer::RenderText(Scene3D* scene, const std::string& text, float x, float y, float scale,
                              const glm::vec4& color) {
  impl_->shader->Bind();

  if (impl_->projection_uniform) {
    ShaderUniforms::SetMat4(impl_->projection_uniform->id, scene->GetCamera()->GetProjectionMatrix());
  }
  if (impl_->color_uniform) {
    ShaderUniforms::SetVec4(impl_->color_uniform->id, color);
  }

  glActiveTexture(GL_TEXTURE0);
  glBindVertexArray(impl_->vao);
  float xscale = scale / static_cast<float>(scene->GetCamera()->GetViewWidth());
  float yscale = scale / static_cast<float>(scene->GetCamera()->GetViewHeight());

  // iterate through all characters
  for (auto c : text) {
    auto& ch = impl_->characters[c];

    float xpos = x + ch->bearing.x * xscale;
    float ypos = y - (ch->size.y - ch->bearing.y) * yscale;

    float w = ch->size.x * xscale;
    float h = ch->size.y * yscale;
    // update vbo for each character
    float vertices[6][4] = {
      {xpos, (ypos + h), 0.0F, 0.0F}, {xpos, ypos, 0.0F, 1.0F},       {(xpos + w), ypos, 1.0F, 1.0F},

      {xpos, (ypos + h), 0.0F, 0.0F}, {(xpos + w), ypos, 1.0F, 1.0F}, {(xpos + w), (ypos + h), 1.0F, 0.0F}};

    // render glyph texture over quad
    glBindTexture(GL_TEXTURE_2D, ch->texture_id);
    // update content of vbo memory
    glBindBuffer(GL_ARRAY_BUFFER, impl_->vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices),
                    vertices);  // be sure to use glBufferSubData and not glBufferData

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // render quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
    // now advance cursors for next glyph (note that advance is number of 1/64 pixels)
    x += static_cast<float>(ch->advance >> 6) * xscale;  // bitshift by 6 to get value in pixels (2^6 = 64 (divide
                                                         // amount of 1/64th pixels by 64 to get amount of pixels))
  }
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  impl_->shader->Unbind();
}
}  // namespace ace_monitor
