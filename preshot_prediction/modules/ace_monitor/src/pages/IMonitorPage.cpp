// Confidential, Copyright 2024, Sony AI, All rights reserved
#include "ace_monitor/IMonitorPage.hpp"

namespace ace_monitor {

const ImVec4 Colors::kNormal = ImVec4(0.5, 1.0, 0.5, 1);
const ImVec4 Colors::kWarning = ImVec4(0.8, 0.4, 0.2, 1);
const ImVec4 Colors::kError = ImVec4(1, 0, 0, 1);
const ImVec4 Colors::kFatal = ImVec4(0.5, 0.1, 0.1, 1);
ImVec4 Colors::ToColor(PageStatus status) noexcept {
  switch (status) {
    case PageStatus::kWarning:
      return Colors::kWarning;
    case PageStatus::kError:
      return Colors::kError;
    case PageStatus::kFatal:
      return Colors::kFatal;
    case PageStatus::kNormal:
      [[fallthrough]];
    default:
      return Colors::kNormal;
  }
}
void IMonitorPage::RenderWidgets() {}

void IMonitorPage::OpenPage() { is_open_ = true; }
void IMonitorPage::ClosePage() { is_open_ = false; }
bool IMonitorPage::DrawPage() {
  if (!is_open_) {
    return false;
  }
  bool is_open = is_open_;

  auto current_color = Colors::ToColor(GetStatus());
  ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, current_color);
  if (ImGui::Begin(page_name_.c_str(), &is_open, ImGuiWindowFlags_AlwaysAutoResize)) {
    // ImGui::PopStyleColor();
    RenderWidgets();

    ImGui::End();
  }
  ImGui::PopStyleColor();
  if (!is_open) {
    ClosePage();
  }
  return is_open;
}

PageStatus IMonitorPage::GetStatus() const { return PageStatus::kNormal; }

}  // namespace ace_monitor
