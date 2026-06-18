// Confidential, Copyright 2024, Sony AI, All rights reserved
#pragma once

#include <imgui.h>

#include <memory>
#include <string>

namespace ace_monitor {

class PerceptionVisualizerPage;
enum class PageStatus { kNormal, kWarning, kError, kFatal };

struct Colors {
  static const ImVec4 kNormal;
  static const ImVec4 kWarning;
  static const ImVec4 kError;
  static const ImVec4 kFatal;

  static ImVec4 ToColor(PageStatus status) noexcept;
};
class IMonitorPage {
 public:
  using SharedPtr = std::shared_ptr<IMonitorPage>;
  using ConstSharedPtr = std::shared_ptr<const IMonitorPage>;

 protected:
  std::string page_name_;
  std::string category_name_;
  bool is_open_{false};

  virtual void RenderWidgets();

 public:
  IMonitorPage(std::string name, std::string category)
    : page_name_(std::move(name)), category_name_(std::move(category)) {}
  virtual ~IMonitorPage() = default;

  [[nodiscard]] const std::string& GetCategoryName() const { return category_name_; }
  [[nodiscard]] const std::string& GetPageName() const { return page_name_; }

  virtual void Initialize(PerceptionVisualizerPage* /*visualizer*/) {}
  virtual void OpenPage();
  virtual void ClosePage();

  virtual void Update(float /*dt*/) {}

  virtual bool DrawPage();

  virtual void Render3D() {}

  [[nodiscard]] virtual PageStatus GetStatus() const;
};
}  // namespace ace_monitor
