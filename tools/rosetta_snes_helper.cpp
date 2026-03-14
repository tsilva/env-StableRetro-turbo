#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "coreinfo.h"
#include "emulator.h"
#include "imageops.h"

namespace {

std::string g_core_dir;

bool read_exact(std::istream* input, void* data, size_t size) {
  input->read(static_cast<char*>(data), static_cast<std::streamsize>(size));
  return input->good();
}

bool write_screen(Retro::Emulator* em) {
  const void* img = em->getImageData();
  if (!img) {
    return false;
  }

  const long width = em->getImageWidth();
  const long height = em->getImageHeight();
  std::vector<uint8_t> rgb(static_cast<size_t>(width * height * 3));
  Retro::Image out(Retro::Image::Format::RGB888, rgb.data(), width, height, width);
  Retro::Image in;

  if (em->getImageDepth() == 16 || em->getImageDepth() == 15) {
    in = Retro::Image(
        Retro::Image::Format::RGB565,
        img,
        width,
        height,
        em->getImagePitch());
  } else if (em->getImageDepth() == 32) {
    in = Retro::Image(
        Retro::Image::Format::RGBX888,
        img,
        width,
        height,
        em->getImagePitch());
  } else {
    return false;
  }

  in.copyTo(&out);
  std::cout << "SCREEN " << width << " " << height << "\n";
  std::cout.write(
      reinterpret_cast<const char*>(rgb.data()),
      static_cast<std::streamsize>(rgb.size()));
  std::cout.flush();
  return true;
}

void write_error(const std::string& message) {
  std::cout << "ERR " << message << "\n";
  std::cout.flush();
}

}  // namespace

namespace Retro {

Variable::Variable(const DataType& type, size_t address, uint64_t mask)
    : type(type), address(address), mask(mask) {}

std::string corePath(const std::string& hint) {
  if (!hint.empty()) {
    g_core_dir = hint;
  }
  return g_core_dir;
}

std::string libForCore(const std::string& core) {
  return core == "Snes" ? "snes9x" : std::string{};
}

std::string coreForRom(const std::string& rom) {
  if (rom.size() >= 4 && rom.substr(rom.size() - 4) == ".sfc") {
    return "Snes";
  }
  if (rom.size() >= 4 && rom.substr(rom.size() - 4) == ".smc") {
    return "Snes";
  }
  return std::string{};
}

std::vector<std::string> buttons(const std::string&) { return {}; }
std::vector<std::string> keybinds(const std::string&) { return {}; }
size_t ramBase(const std::string&) { return 0; }
void configureData(GameData*, const std::string&) {}
bool loadCoreInfo(const std::string&) { return true; }
std::vector<std::string> cores() { return {}; }
std::vector<std::string> extensions() { return {}; }

}  // namespace Retro

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: rosetta_snes_helper <core_dir> <rom_path>\n";
    return 2;
  }

  const std::string core_dir = argv[1];
  const std::string rom_path = argv[2];

  g_core_dir = core_dir;
  setenv("RETRO_CORE_PATH", core_dir.c_str(), 1);

  Retro::Emulator em;
  if (!em.loadRom(rom_path)) {
    std::cerr << "failed to load rom\n";
    return 4;
  }
  em.run();

  std::cout << "READY\n";
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream cmd(line);
    std::string op;
    cmd >> op;

    if (op == "PING") {
      std::cout << "OK\n";
      std::cout.flush();
      continue;
    }

    if (op == "QUIT") {
      std::cout << "OK\n";
      std::cout.flush();
      return 0;
    }

    if (op == "STEP") {
      em.run();
      std::cout << "OK\n";
      std::cout.flush();
      continue;
    }

    if (op == "BUTTON") {
      unsigned player = 0;
      uint64_t mask = 0;
      cmd >> player >> mask;
      for (int key = 0; key < Retro::N_BUTTONS; ++key) {
        em.setKey(player, key, (mask >> key) & 1U);
      }
      std::cout << "OK\n";
      std::cout.flush();
      continue;
    }

    if (op == "STATE") {
      size_t size = 0;
      cmd >> size;
      std::vector<uint8_t> state(size);
      if (!read_exact(&std::cin, state.data(), state.size())) {
        write_error("state-read-failed");
        return 5;
      }
      int term = std::cin.get();
      if (term != '\n') {
        write_error("state-terminator-missing");
        return 6;
      }
      if (!em.unserialize(state.data(), state.size())) {
        write_error("state-load-failed");
        continue;
      }
      std::cout << "OK\n";
      std::cout.flush();
      continue;
    }

    if (op == "SCREEN") {
      if (!write_screen(&em)) {
        write_error("screen-unavailable");
      }
      continue;
    }

    if (op == "CLEAR_CHEATS") {
      em.clearCheats();
      std::cout << "OK\n";
      std::cout.flush();
      continue;
    }

    if (op == "ADD_CHEAT") {
      size_t size = 0;
      cmd >> size;
      std::vector<char> cheat(size);
      if (!read_exact(&std::cin, cheat.data(), cheat.size())) {
        write_error("cheat-read-failed");
        return 7;
      }
      int term = std::cin.get();
      if (term != '\n') {
        write_error("cheat-terminator-missing");
        return 8;
      }
      cheat.push_back('\0');
      em.setCheat(0, true, cheat.data());
      std::cout << "OK\n";
      std::cout.flush();
      continue;
    }

    write_error("unknown-command");
  }

  return 0;
}
