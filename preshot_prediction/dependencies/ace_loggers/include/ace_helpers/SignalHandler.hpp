// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <csignal>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace utils {

class SignalHandler {
 private:
  SignalHandler() = default;
  ~SignalHandler();

  std::map<int, std::map<std::string, std::function<void(int)>>> handlers_;

  static SignalHandler instance_;
  static void SignalHandlerCB(int signal);

 public:
  static bool AddSignalHandler(const std::string& id, int signal, std::function<void(int)> handler);
  static bool RemoveSignalHandler(const std::string& id, int signal);
  static void RaiseSignal(int signal);
};

}  // namespace utils
