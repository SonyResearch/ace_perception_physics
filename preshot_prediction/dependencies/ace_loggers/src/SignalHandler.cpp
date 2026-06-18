#include "ace_helpers/SignalHandler.hpp"

#include <iostream>

namespace utils {
SignalHandler SignalHandler::instance_;
SignalHandler::~SignalHandler() { instance_.handlers_.clear(); }

bool SignalHandler::AddSignalHandler(const std::string& id, int signal, std::function<void(int)> handler) {
  if (instance_.handlers_.find(signal) == instance_.handlers_.end()) {
    instance_.handlers_[signal] = std::map<std::string, std::function<void(int)>>();
    std::signal(signal, SignalHandler::SignalHandlerCB);
  }
  if (instance_.handlers_[signal].find(id) != instance_.handlers_[signal].end()) {
    return false;
  }
  instance_.handlers_[signal][id] = handler;
  return true;
}

bool SignalHandler::RemoveSignalHandler(const std::string& id, int signal) {
  if (instance_.handlers_.find(signal) == instance_.handlers_.end()) {
    return false;
  }
  auto it = instance_.handlers_[signal].find(id);
  if (it != instance_.handlers_[signal].end()) {
    return false;
  }
  instance_.handlers_[signal].erase(it);
  return true;
}

void SignalHandler::SignalHandlerCB(int signal) {
  std::cout << "Received exit signal: " << signal << std::endl;
  if (instance_.handlers_.find(signal) != instance_.handlers_.end()) {
    for (const auto& [id, handler] : instance_.handlers_[signal]) {
      handler(signal);
    }
  }
  exit(signal);
}

void SignalHandler::RaiseSignal(int signal) { std::raise(signal); }
}  // namespace utils
