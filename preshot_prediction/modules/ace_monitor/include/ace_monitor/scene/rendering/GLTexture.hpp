// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once
#include <memory>
#include <string>

namespace ace_monitor {
class GLTexture {
 public:
  using SharedPtr = std::shared_ptr<GLTexture>;

 private:
  unsigned int texture_id_{0};
  std::string name_;

 public:
  explicit GLTexture(std::string name);
  ~GLTexture();

  void Destroy();

  [[nodiscard]] const std::string& GetName() const { return name_; }

  void SetTextureID(unsigned int id) { texture_id_ = id; }
  [[nodiscard]] unsigned int GetTextureID() const { return texture_id_; }

  void Bind(int target = 0) const;
  static void Unbind(int target = 0);
};

}  // namespace ace_monitor
