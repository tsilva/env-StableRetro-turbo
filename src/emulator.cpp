#include <cassert>
#include <cmath>
#include <cstdlib>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include "coreinfo.h"
#include "data.h"
#include "emulator.h"
#include "libretro.h"

#ifndef _WIN32
#define GETSYM dlsym
#else
#define GETSYM GetProcAddress
#endif

using namespace std;

namespace Retro {

static Emulator* s_loadedEmulator = nullptr;
static thread_local Emulator* s_callbackEmulator = nullptr;
static std::mutex s_emulatorMutex;
static int s_loadedEmulatorCount = 0;

static map<string, string> s_envVariables = {
	{ "genesis_plus_gx_bram", "per game" },
	{ "genesis_plus_gx_render", "single field" },
	{ "genesis_plus_gx_blargg_ntsc_filter", "disabled" },

	// Parallel-N64 defaults: force a CPU-rendered framebuffer so the frontend can read pixels.
	// If left on auto, the core may choose an OpenGL path and provide no CPU buffer.
	{ "parallel-n64-gfxplugin", "angrylion" },

	// Beetle Saturn: trim overscan so hi-res video fills the viewport without letterboxing.
	{ "beetle_saturn_initial_scanline", "8" },
	{ "beetle_saturn_last_scanline", "231" },
	{ "beetle_saturn_initial_scanline_pal", "16" },
	{ "beetle_saturn_last_scanline_pal", "271" },

	// Flycast (Dreamcast) settings for training
	{ "flycast_internal_resolution", "640x480" },
	{ "flycast_gdrom_fast_loading", "enabled" },
	{ "flycast_system", "auto" },
	{ "flycast_boot_to_bios", "disabled" },
	{ "flycast_mipmapping", "disabled" },
	{ "flycast_volume_modifier_enable", "disabled" },
	{ "flycast_threaded_rendering", "enabled" },
	{ "flycast_alpha_sorting", "per-strip (fast, least accurate)" },
	// Force OpenGL instead of Vulkan
	{ "flycast_pvr2_filtering", "disabled" },
	{ "flycast_region", "Auto" },

	// melonDS (Nintendo DS) settings for training
	{ "melonds_boot_directly", "enabled" },
	{ "melonds_console_mode", "DS" },
	{ "melonds_threaded_renderer", "enabled" },
	{ "melonds_opengl_renderer", "disabled" },
	{ "melonds_jit_enable", "enabled" },
	{ "melonds_touch_mode", "joystick" },
	{ "melonds_screen_layout", "Bottom Only" },
};

static bool envFlagEnabled(const char* name) {
	const char* value = getenv(name);
	if (!value || !*value) {
		return false;
	}
	return strcmp(value, "0") && strcmp(value, "false") && strcmp(value, "False") && strcmp(value, "FALSE");
}

class CallbackScope {
public:
	explicit CallbackScope(Emulator* emulator)
		: m_previous(s_callbackEmulator) {
		s_callbackEmulator = emulator;
	}
	~CallbackScope() {
		s_callbackEmulator = m_previous;
	}
private:
	Emulator* m_previous;
};

static Emulator* callbackEmulator() {
	Emulator* emulator = s_callbackEmulator ? s_callbackEmulator : s_loadedEmulator;
	assert(emulator);
	return emulator;
}

static string copyCoreForIsolation(const string& corePath) {
#ifdef _WIN32
	return "";
#else
	string suffix;
#ifdef __APPLE__
	suffix = ".dylib";
#else
	suffix = ".so";
#endif
	string tmpl = "/tmp/stable-retro-core-XXXXXX" + suffix;
	vector<char> path(tmpl.begin(), tmpl.end());
	path.push_back('\0');
	int fd = mkstemps(path.data(), static_cast<int>(suffix.size()));
	if (fd < 0) {
		return "";
	}

	ifstream in(corePath, ios::binary);
	if (!in) {
		close(fd);
		unlink(path.data());
		return "";
	}
	vector<char> buffer(1024 * 1024);
	while (in) {
		in.read(buffer.data(), buffer.size());
		streamsize count = in.gcount();
		if (count > 0) {
			const char* data = buffer.data();
			while (count > 0) {
				ssize_t written = write(fd, data, static_cast<size_t>(count));
				if (written <= 0) {
					close(fd);
					unlink(path.data());
					return "";
				}
				data += written;
				count -= written;
			}
		}
	}
	close(fd);
	return string(path.data());
#endif
}

Emulator::Emulator() {
	m_audioEnabled = !envFlagEnabled("STABLE_RETRO_DISABLE_AUDIO");
}

Emulator::~Emulator() {
	if (m_corePath) {
		free(m_corePath);
	}
	if (m_romLoaded) {
		unloadRom();
	}
#ifdef ENABLE_HW_RENDER
	// Destroy HW render context after unloading ROM but before unloading core
	// This ensures the GL context is still valid during context_destroy callback
	m_hwRender.destroy();
#endif
	if (m_coreHandle) {
		unloadCore();
	}
}

bool Emulator::isLoaded() {
	std::lock_guard<std::mutex> lock(s_emulatorMutex);
	return s_loadedEmulatorCount > 0;
}

bool Emulator::loadRom(const string& romPath) {
	if (m_romLoaded) {
		unloadRom();
	}
	if (romPath.empty()) {
		return false;
	}

	auto core = coreForRom(romPath);
	if (core.size() == 0) {
		return false;
	}

	if (m_coreHandle && m_core != core) {
		unloadCore();
	}
	if (!m_coreHandle) {
		string lib = libForCore(core) + "_libretro.";
#ifdef __APPLE__
		lib += "dylib";
#elif defined(_WIN32)
		lib += "dll";
#else
		lib += "so";
#endif
		string fullPath = corePath() + "/" + lib;
		if (!loadCore(fullPath)) {
			return false;
		}
		m_core = core;
	}

	m_romPath = romPath;
	ifstream in(romPath, ios::binary | ios::ate);
	if (in.fail()) {
		return false;
	}
	ostringstream out;
	m_gameInfo.size = in.tellg();
	if (in.fail()) {
		return false;
	}
	m_romData.resize(m_gameInfo.size);
	m_gameInfo.path = m_romPath.c_str();
	m_gameInfo.data = m_romData.data();
	in.seekg(0, ios::beg);
	in.read(m_romData.data(), m_gameInfo.size);
	if (in.fail()) {
		m_romData.clear();
		return false;
	}
	in.close();

	m_rotation = 0;
	CallbackScope callbackScope(this);
	auto res = m_retro_load_game(&m_gameInfo);
	if (!res) {
		m_romData.clear();
		return false;
	}

	retro_system_info systemInfo;
	m_retro_get_system_info(&systemInfo);
	if (!strcmp(systemInfo.library_name, "Snes9x")) {
		// Snes9x expects an explicit reset after load_game on Apple Silicon.
		// Without it, cold boot never produces frames and savestates collapse
		// after a few steps because core state is only partially initialized.
		m_retro_reset();
	}

#ifdef ENABLE_HW_RENDER
	// If HW rendering was enabled during m_retro_load_game, now call context_reset
	// This must be done after RETRO_ENVIRONMENT_SET_HW_RENDER has returned and
	// the frontend callbacks are wired up
	if (m_hwRender.isEnabled()) {
		std::cerr << "Emulator: About to call m_hwRender.contextReset()..." << std::endl;
		m_hwRender.contextReset();
		std::cerr << "Emulator: m_hwRender.contextReset() returned" << std::endl;
	}
#endif

	m_retro_get_system_av_info(&m_avInfo);
	fixScreenSize(romPath);

	// For some cores (notably some N64 cores), the initial AV info can be wrong.
	// Prefer the per-frame dimensions passed to cbVideoRefresh.
	{
		m_updateGeometryFromVideoRefresh =
			!strcmp(systemInfo.library_name, "ParaLLEl N64") ||
			!strcmp(systemInfo.library_name, "Mupen64Plus") ||
			!strcmp(systemInfo.library_name, "Mupen64Plus-Next") ||
			!strcmp(systemInfo.library_name, "Beetle Saturn") ||
			!strcmp(systemInfo.library_name, "Mednafen Saturn");
	}

	if (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) {
		m_needsInitFrame = true;
	}

	m_romLoaded = true;
	m_romPath = romPath;
	return true;
}

void Emulator::run() {
	if (m_audioEnabled) {
		m_audioData.clear();
	}
	CallbackScope callbackScope(this);
	m_retro_run();
	if (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) {
		m_needsInitFrame = false;
	}
}

bool Emulator::runSkipRender() {
	if (!m_stable_retro_run_skip_render) {
		run();
		return false;
	}
	if (m_audioEnabled) {
		m_audioData.clear();
	}
	CallbackScope callbackScope(this);
	m_stable_retro_run_skip_render();
	if (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) {
		m_needsInitFrame = false;
	}
	return true;
}

void Emulator::reset() {
	memset(m_buttonMask, 0, sizeof(m_buttonMask));

	CallbackScope callbackScope(this);
	retro_system_info systemInfo;
	m_retro_get_system_info(&systemInfo);
	if (!strcmp(systemInfo.library_name, "Stella")) {
		// Stella does not properly clear everything when reseting or loading a savestate
		string romPath = m_romPath;

#ifdef _WIN32
		FreeLibrary(m_coreHandle);
#else
		dlclose(m_coreHandle);
#endif
		m_coreHandle = nullptr;
		{
			std::lock_guard<std::mutex> lock(s_emulatorMutex);
			if (s_loadedEmulator == this) {
				s_loadedEmulator = nullptr;
			}
			if (s_loadedEmulatorCount > 0) {
				--s_loadedEmulatorCount;
			}
		}
		m_romLoaded = false;
		loadRom(m_romPath);
		if (m_addressSpace) {
			m_addressSpace->reset();
			m_addressSpace->addBlock(Retro::ramBase(m_core), m_retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM), m_retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM));
		}
	}

	m_retro_reset();

	if (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) {
		m_needsInitFrame = true;
	}
}

void Emulator::unloadCore() {
	if (!m_coreHandle) {
		return;
	}
	if (m_romLoaded) {
		unloadRom();
	}
	CallbackScope callbackScope(this);
	m_retro_deinit();
#ifdef _WIN32
	FreeLibrary(m_coreHandle);
#else
	dlclose(m_coreHandle);
#endif
	m_coreHandle = nullptr;
	{
		std::lock_guard<std::mutex> lock(s_emulatorMutex);
		if (s_loadedEmulator == this) {
			s_loadedEmulator = nullptr;
		}
		if (s_loadedEmulatorCount > 0) {
			--s_loadedEmulatorCount;
		}
	}
	if (!m_coreCopyPath.empty()) {
#ifndef _WIN32
		unlink(m_coreCopyPath.c_str());
#endif
		m_coreCopyPath.clear();
	}
}

void Emulator::unloadRom() {
	if (!m_romLoaded) {
		return;
	}
	CallbackScope callbackScope(this);
	m_retro_unload_game();
	m_romLoaded = false;
	m_romPath.clear();
	m_romData.clear();
	m_gameInfo = {};
	m_addressSpace = nullptr;
	m_map.clear();
}

bool Emulator::serialize(void* data, size_t size) {
	ensureInitializedForSerialization();
	CallbackScope callbackScope(this);
	return m_retro_serialize(data, size);
}

bool Emulator::unserialize(const void* data, size_t size) {
	try {
		CallbackScope callbackScope(this);
		retro_system_info systemInfo;
		m_retro_get_system_info(&systemInfo);
		if (!strcmp(systemInfo.library_name, "Stella")) {
			// Match Stable Retro 1.0.1 exactly: Stella does not clear every
			// device-local field when a saved state is loaded, so rebuild the core
			// before applying the authority-format state.
			reset();
		}
		ensureInitializedForSerialization();
		bool ok = m_retro_unserialize(data, size);
		if (ok && (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE)) {
			m_needsInitFrame = false;
		}
		return ok;
	} catch (...) {
		return false;
	}
}

bool Emulator::unserializeLive(const void* data, size_t size) {
	try {
		CallbackScope callbackScope(this);
		ensureInitializedForSerialization();
		const bool ok = m_retro_unserialize(data, size);
		if (ok && (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE)) {
			m_needsInitFrame = false;
		}
		return ok;
	} catch (...) {
		return false;
	}
}

size_t Emulator::serializeSize() {
	CallbackScope callbackScope(this);
	return m_retro_serialize_size();
}

void Emulator::ensureInitializedForSerialization() {
	if ((m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) && m_needsInitFrame) {
		// Run a single frame to satisfy cores that require initialization before (de)serialization
		run();
	}
}

void Emulator::clearCheats() {
	CallbackScope callbackScope(this);
	m_retro_cheat_reset();
}

void Emulator::setCheat(unsigned index, bool enabled, const char* code) {
	CallbackScope callbackScope(this);
	m_retro_cheat_set(index, enabled, code);
}

bool Emulator::loadCore(const string& corePath) {
	string loadPath = corePath;
	{
		std::lock_guard<std::mutex> lock(s_emulatorMutex);
		if (s_loadedEmulatorCount > 0) {
			m_coreCopyPath = copyCoreForIsolation(corePath);
			if (!m_coreCopyPath.empty()) {
				loadPath = m_coreCopyPath;
			}
		}
	}

#ifdef _WIN32
	m_coreHandle = LoadLibrary(loadPath.c_str());
#else
	m_coreHandle = dlopen(loadPath.c_str(), RTLD_LAZY);
#endif
	if (!m_coreHandle) {
		if (!m_coreCopyPath.empty()) {
#ifndef _WIN32
			unlink(m_coreCopyPath.c_str());
#endif
			m_coreCopyPath.clear();
		}
		return false;
	}

	m_retro_init = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "retro_init"));
	m_retro_deinit = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "retro_deinit"));
	m_retro_api_version = reinterpret_cast<unsigned int (*)()>(GETSYM(m_coreHandle, "retro_api_version"));
	m_retro_get_system_info = reinterpret_cast<void (*)(struct retro_system_info*)>(GETSYM(m_coreHandle, "retro_get_system_info"));
	m_retro_get_system_av_info = reinterpret_cast<void (*)(struct retro_system_av_info*)>(GETSYM(m_coreHandle, "retro_get_system_av_info"));
	m_retro_reset = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "retro_reset"));
	m_retro_run = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "retro_run"));
	m_stable_retro_run_skip_render = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "stable_retro_run_skip_render"));
	m_retro_serialize_size = reinterpret_cast<size_t (*)()>(GETSYM(m_coreHandle, "retro_serialize_size"));
	m_retro_serialize = reinterpret_cast<bool (*)(void*, size_t)>(GETSYM(m_coreHandle, "retro_serialize"));
	m_retro_unserialize = reinterpret_cast<bool (*)(const void*, size_t)>(GETSYM(m_coreHandle, "retro_unserialize"));
	m_retro_load_game = reinterpret_cast<bool (*)(const struct retro_game_info*)>(GETSYM(m_coreHandle, "retro_load_game"));
	m_retro_unload_game = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "retro_unload_game"));
	m_retro_get_memory_data = reinterpret_cast<void* (*) (unsigned int)>(GETSYM(m_coreHandle, "retro_get_memory_data"));
	m_retro_get_memory_size = reinterpret_cast<size_t (*)(unsigned int)>(GETSYM(m_coreHandle, "retro_get_memory_size"));
	m_retro_cheat_reset = reinterpret_cast<void (*)()>(GETSYM(m_coreHandle, "retro_cheat_reset"));
	m_retro_cheat_set = reinterpret_cast<void (*)(unsigned int, bool, const char*)>(GETSYM(m_coreHandle, "retro_cheat_set"));
	m_retro_set_environment = reinterpret_cast<void (*)(retro_environment_t)>(GETSYM(m_coreHandle, "retro_set_environment"));
	m_retro_set_video_refresh = reinterpret_cast<void (*)(retro_video_refresh_t)>(GETSYM(m_coreHandle, "retro_set_video_refresh"));
	m_retro_set_audio_sample = reinterpret_cast<void (*)(retro_audio_sample_t)>(GETSYM(m_coreHandle, "retro_set_audio_sample"));
	m_retro_set_audio_sample_batch = reinterpret_cast<void (*)(retro_audio_sample_batch_t)>(GETSYM(m_coreHandle, "retro_set_audio_sample_batch"));
	m_retro_set_input_poll = reinterpret_cast<void (*)(retro_input_poll_t)>(GETSYM(m_coreHandle, "retro_set_input_poll"));
	m_retro_set_input_state = reinterpret_cast<void (*)(short (*)(unsigned int, unsigned int, unsigned int, unsigned int))>(GETSYM(m_coreHandle, "retro_set_input_state"));
	m_stable_retro_set_audio_enabled = reinterpret_cast<void (*)(bool)>(GETSYM(m_coreHandle, "stable_retro_set_audio_enabled"));
	m_stable_retro_set_indexed_video = reinterpret_cast<void (*)(bool)>(GETSYM(m_coreHandle, "stable_retro_set_indexed_video"));
	m_stable_retro_get_indexed_video = reinterpret_cast<bool (*)(const uint8_t**, const uint16_t**, unsigned*, unsigned*, size_t*, bool*, int*)>(GETSYM(m_coreHandle, "stable_retro_get_indexed_video"));
	m_stable_retro_get_previous_indexed_video = reinterpret_cast<bool (*)(const uint8_t**)>(GETSYM(m_coreHandle, "stable_retro_get_previous_indexed_video"));

	// The default according to the docs
	m_imgDepth = 15;
	{
		std::lock_guard<std::mutex> lock(s_emulatorMutex);
		if (!s_loadedEmulator) {
			s_loadedEmulator = this;
		}
		++s_loadedEmulatorCount;
	}

	CallbackScope callbackScope(this);
	m_retro_set_environment(cbEnvironment);
	m_retro_set_video_refresh(cbVideoRefresh);
	m_retro_set_audio_sample(cbAudioSample);
	m_retro_set_audio_sample_batch(cbAudioSampleBatch);
	m_retro_set_input_poll(cbInputPoll);
	m_retro_set_input_state(cbInputState);
	m_retro_init();
	if (m_stable_retro_set_audio_enabled) {
		m_stable_retro_set_audio_enabled(m_audioEnabled);
	}

		if (m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) {
			m_needsInitFrame = true;
		}

	return true;
}

void Emulator::setAudioEnabled(bool enabled) {
	m_audioEnabled = enabled;
	if (m_stable_retro_set_audio_enabled) {
		m_stable_retro_set_audio_enabled(enabled);
	}
	if (!enabled) {
		m_audioData.clear();
	}
}

bool Emulator::setIndexedVideoEnabled(bool enabled) {
	if (!m_stable_retro_set_indexed_video || !m_stable_retro_get_indexed_video) {
		return false;
	}
	m_stable_retro_set_indexed_video(enabled);
	return true;
}

bool Emulator::getIndexedVideoFrame(IndexedVideoFrame& frame) {
	if (!m_stable_retro_get_indexed_video) {
		return false;
	}
	frame.previousData = nullptr;
	const bool valid = m_stable_retro_get_indexed_video(
		&frame.data,
		&frame.palette,
		&frame.width,
		&frame.height,
		&frame.pitch,
		&frame.rawPalette,
		&frame.deemp
	) && frame.data && frame.palette && frame.width && frame.height && frame.pitch;
	if (valid && m_stable_retro_get_previous_indexed_video) {
		m_stable_retro_get_previous_indexed_video(&frame.previousData);
	}
	return valid;
}

double Emulator::getAspectRatio() const {
	double ratio = m_avInfo.geometry.aspect_ratio;
	if (!std::isfinite(ratio) || ratio <= 0.0) {
		const unsigned width = m_avInfo.geometry.base_width;
		const unsigned height = m_avInfo.geometry.base_height;
		ratio = height ? static_cast<double>(width) / static_cast<double>(height) : 1.0;
	}
	if ((m_rotation & 1) != 0) {
		ratio = 1.0 / ratio;
	}
	return ratio;
}

void Emulator::fixScreenSize(const string& romName) {
	CallbackScope callbackScope(this);
	retro_system_info systemInfo;
	m_retro_get_system_info(&systemInfo);
	if (!strcmp(systemInfo.library_name, "Genesis Plus GX")) {
		switch (romName.back()) {
		case 'd': // Mega Drive
			// Genesis Plus GX gives us too small a resolution initially
			m_avInfo.geometry.base_width = 320;
			m_avInfo.geometry.base_height = 224;
			break;
		case 's': // Master System
			// Genesis Plus GX gives us too small a resolution initially
			m_avInfo.geometry.base_width = 256;
			m_avInfo.geometry.base_height = 192;
			break;
		case 'g': // Game Gear
			m_avInfo.geometry.base_width = 160;
			m_avInfo.geometry.base_height = 144;
			break;
		}
	} else if (!strcmp(systemInfo.library_name, "Stella")) {
		// Stella gives confusing values to pretend the pixel width is 2x
		m_avInfo.geometry.base_width = 160;
	} else if (!strcmp(systemInfo.library_name, "Mednafen PCE Fast")) {
		m_avInfo.geometry.base_width = 256;
		m_avInfo.geometry.base_height = 242;
	} else if (!strcmp(systemInfo.library_name, "Beetle Saturn") ||
			   !strcmp(systemInfo.library_name, "Mednafen Saturn")) {
		// Beetle Saturn always reports 320x240 even when outputting wider/higher frames
		m_avInfo.geometry.base_width = 352;
		m_avInfo.geometry.base_height = 240;
		m_updateGeometryFromVideoRefresh = true;
	} else if (!strcmp(systemInfo.library_name, "ParaLLEl N64") ||
			   !strcmp(systemInfo.library_name, "Mupen64Plus") ||
			   !strcmp(systemInfo.library_name, "Mupen64Plus-Next")) {
		// Some N64 libretro cores report a half-height (or otherwise unexpected)
		// base_height which causes the frontend to display only the top half
		// of the frame. Ensure we have at least a 480px height reported so the
		// image isn't vertically cropped.
		if (m_avInfo.geometry.base_height < 480) {
			m_avInfo.geometry.base_height = 480;
		}
	}
}

void Emulator::reconfigureAddressSpace() {
	if (!m_addressSpace) {
		return;
	}
	if (!m_map.empty() && m_addressSpace->blocks().empty()) {
		for (const auto& desc : m_map) {
			if (desc.flags & RETRO_MEMDESC_CONST) {
				continue;
			}
			size_t len = desc.len;
			if (desc.select) {
				len = ((~desc.select & ~desc.start) + 1) & desc.select;
			}
			if (desc.len && desc.len < len) {
				len = desc.len;
			}
			m_addressSpace->addBlock(desc.start, len, desc.ptr);
		}
	}
}

// callback for logging from emulator
// turned off by default to avoid spamming the log, only used for debugging issues within cores
static void cbLog(enum retro_log_level level, const char *fmt, ...) {
#if 0
	char buffer[4096] = {0};
	static const char * levelName[] = { "DEBUG", "INFO", "WARNING", "ERROR" };
	va_list va;

	va_start(va, fmt);
	std::vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	if (level == 0)
		return;

	std::cerr << "[" << levelName[level] << "] " << buffer << std::flush;
#endif
}

bool Emulator::cbEnvironment(unsigned cmd, void* data) {
	Emulator* emulator = callbackEmulator();

	switch (cmd) {
	case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
		switch (*reinterpret_cast<retro_pixel_format*>(data)) {
		case RETRO_PIXEL_FORMAT_XRGB8888:
			emulator->m_imgDepth = 32;
			break;
		case RETRO_PIXEL_FORMAT_RGB565:
			emulator->m_imgDepth = 16;
			break;
		case RETRO_PIXEL_FORMAT_0RGB1555:
			emulator->m_imgDepth = 15;
			break;
		default:
			emulator->m_imgDepth = 0;
			break;
		}
		return true;
	case RETRO_ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER: {
		// Stable Retro 1.0.1 did not offer this frontend fast path.  FCEUmm's
		// behavior is not equivalent when it is enabled: on some NES frames it
		// presents a newly cleared frontend buffer instead of duplicating the
		// last core-owned frame.  Keep NES on the authority's public video path.
		if (emulator->m_core == "Nes") {
			return false;
		}
		auto* framebuffer = reinterpret_cast<retro_framebuffer*>(data);
		if (!framebuffer || !framebuffer->width || !framebuffer->height) {
			return false;
		}
		size_t bytesPerPixel = 0;
		switch (emulator->m_imgDepth) {
		case 16:
			bytesPerPixel = 2;
			framebuffer->format = RETRO_PIXEL_FORMAT_RGB565;
			break;
		case 32:
			bytesPerPixel = 4;
			framebuffer->format = RETRO_PIXEL_FORMAT_XRGB8888;
			break;
		default:
			return false;
		}
		const size_t rowBytes = static_cast<size_t>(framebuffer->width) * bytesPerPixel;
		const size_t bytes = rowBytes * static_cast<size_t>(framebuffer->height);
		emulator->m_softwareFramebuffer.resize(bytes);
		framebuffer->data = emulator->m_softwareFramebuffer.data();
		framebuffer->pitch = rowBytes;
		framebuffer->memory_flags = 0;
		return true;
	}
	case RETRO_ENVIRONMENT_SET_VARIABLES: {
		auto* vars = reinterpret_cast<const retro_variable*>(data);
		if (!vars) {
			return false;
		}
		for (; vars->key; ++vars) {
			if (!vars->value || s_envVariables.count(vars->key)) {
				continue;
			}
			const char* defaults = strchr(vars->value, ';');
			if (!defaults) {
				continue;
			}
			defaults += 1;
			while (*defaults == ' ') {
				++defaults;
			}
			const char* end = strchr(defaults, '|');
			s_envVariables[vars->key] =
				end ? string(defaults, end - defaults) : string(defaults);
		}
		return true;
	}
	case RETRO_ENVIRONMENT_GET_VARIABLE: {
		struct retro_variable* var = reinterpret_cast<struct retro_variable*>(data);
		if (s_envVariables.count(string(var->key))) {
			var->value = s_envVariables[string(var->key)].c_str();
			return true;
		}
		return false;
	}
	case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
		*reinterpret_cast<bool*>(data) = false;
		return true;
	case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
	case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
		if (!emulator->m_corePath) {
			emulator->m_corePath = strdup(corePath().c_str());
		}
		*reinterpret_cast<const char**>(data) = emulator->m_corePath;
		return true;
	case RETRO_ENVIRONMENT_GET_CAN_DUPE:
		*reinterpret_cast<bool*>(data) = true;
		return true;
	case RETRO_ENVIRONMENT_SET_MEMORY_MAPS:
		emulator->m_map.clear();
		for (size_t i = 0; i < static_cast<const retro_memory_map*>(data)->num_descriptors; ++i) {
			emulator->m_map.emplace_back(static_cast<const retro_memory_map*>(data)->descriptors[i]);
		}
		emulator->reconfigureAddressSpace();
		return true;
	case RETRO_ENVIRONMENT_SET_GEOMETRY: {
		auto* geom = reinterpret_cast<const retro_system_av_info*>(data);
		if (geom) {
			emulator->m_avInfo.geometry = geom->geometry;
			emulator->m_avInfo.timing = geom->timing;
		}
		return true;
	}
	case RETRO_ENVIRONMENT_SET_CONTROLLER_INFO:
	case RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS:
	case RETRO_ENVIRONMENT_SET_PERFORMANCE_LEVEL:
	case RETRO_ENVIRONMENT_SET_SUPPORT_ACHIEVEMENTS:
		return true;
	case RETRO_ENVIRONMENT_SET_ROTATION: {
		const unsigned* rotation = reinterpret_cast<const unsigned*>(data);
		if (rotation) {
			unsigned raw = *rotation % 4;
			if (emulator->m_core == "FBNeo") {
				raw = (4 - raw) % 4;
			}
			emulator->m_rotation = static_cast<int>(raw);
		}
		return true;
	}
	// Logs needs to be handled even when not used, otherwise some cores (ex: mame2003_plus) will crash
	// Also very useful when integrating new emulators to debug issues within the core itself
	case RETRO_ENVIRONMENT_GET_LOG_INTERFACE: {
		struct retro_log_callback *cb = (struct retro_log_callback *)data;
		cb->log = cbLog;
		return true;
	}
	case RETRO_ENVIRONMENT_SET_SERIALIZATION_QUIRKS: {
		emulator->m_serializationQuirks = *reinterpret_cast<const uint64_t*>(data);
		if (emulator->m_serializationQuirks & RETRO_SERIALIZATION_QUIRK_MUST_INITIALIZE) {
			emulator->m_needsInitFrame = true;
		}
		return true;
	}
#ifdef ENABLE_HW_RENDER
	case RETRO_ENVIRONMENT_SET_HW_RENDER: {
		auto* cb = static_cast<retro_hw_render_callback*>(data);
		if (!emulator->m_hwRender.init(*cb)) {
			return false;
		}
		// Provide frontend callbacks to the core
		cb->get_current_framebuffer = cbGetCurrentFramebuffer;
		cb->get_proc_address = cbGetProcAddress;
		return true;
	}
#endif
	default:
		return false;
	}
	return false;
}

void Emulator::cbVideoRefresh(const void* data, unsigned width, unsigned height, size_t pitch) {
	Emulator* emulator = callbackEmulator();
	if (emulator->m_updateGeometryFromVideoRefresh && width && height) {
		emulator->m_avInfo.geometry.base_width = width;
		emulator->m_avInfo.geometry.base_height = height;

		emulator->m_avInfo.geometry.aspect_ratio =
			static_cast<float>(width) / static_cast<float>(height);

		if (emulator->m_avInfo.geometry.max_width < width) {
			emulator->m_avInfo.geometry.max_width = width;
		}
		if (emulator->m_avInfo.geometry.max_height < height) {
			emulator->m_avInfo.geometry.max_height = height;
		}
	}
	// Hardware rendering: the core is signaling that the framebuffer lives on the GPU.
	if (data == RETRO_HW_FRAME_BUFFER_VALID) {
#ifdef ENABLE_HW_RENDER
		if (emulator->m_hwRender.isEnabled()) {
			// Read pixels from GPU framebuffer to CPU
			const void* pixels = emulator->m_hwRender.readbackFramebuffer(width, height);
			if (pixels) {
				emulator->m_imgDepth = 32;  // RGBA8888
				data = pixels;
				pitch = emulator->m_hwRender.getReadbackPitch();
			} else {
				return;
			}
		}
#else
		return;
#endif
	}

	// NULL means "duplicate the previous frame".
	if (!data || !width || !height) {
		return;
	}

	size_t bytes_per_pixel = 0;
	switch (emulator->m_imgDepth) {
	case 15:
	case 16:
		bytes_per_pixel = 2;
		break;
	case 32:
		bytes_per_pixel = 4;
		break;
	default:
		return;
	}

	const size_t row_bytes = width * bytes_per_pixel;
	if (!pitch) {
		pitch = row_bytes;
	}
	if (pitch < row_bytes) {
		if (data == emulator->m_softwareFramebuffer.data()) {
			pitch = row_bytes;
		} else {
			return;
		}
	}

	if (data == emulator->m_softwareFramebuffer.data() && pitch == row_bytes) {
		emulator->m_imgData = data;
		emulator->m_imgPitch = row_bytes;
		return;
	}

	emulator->m_imgBuffer.resize(row_bytes * height);
	auto* dst = emulator->m_imgBuffer.data();
	const auto* src = static_cast<const uint8_t*>(data);
	for (unsigned y = 0; y < height; ++y) {
		memcpy(dst + (y * row_bytes), src + (y * pitch), row_bytes);
	}

	emulator->m_imgData = dst;
	emulator->m_imgPitch = row_bytes;
}

void Emulator::cbAudioSample(int16_t left, int16_t right) {
	Emulator* emulator = callbackEmulator();
	if (!emulator->m_audioEnabled) {
		return;
	}
	emulator->m_audioData.push_back(left);
	emulator->m_audioData.push_back(right);
}

size_t Emulator::cbAudioSampleBatch(const int16_t* data, size_t frames) {
	Emulator* emulator = callbackEmulator();
	if (!emulator->m_audioEnabled) {
		return frames;
	}
	emulator->m_audioData.insert(emulator->m_audioData.end(), data, &data[frames * 2]);
	return frames;
}

void Emulator::cbInputPoll() {
	callbackEmulator();
}

int16_t Emulator::cbInputState(unsigned port, unsigned, unsigned, unsigned id) {
	Emulator* emulator = callbackEmulator();
	return emulator->m_buttonMask[port][id];
}

void Emulator::configureData(GameData* data) {
	m_addressSpace = &data->addressSpace();
	m_addressSpace->reset();
	Retro::configureData(data, m_core);
	reconfigureAddressSpace();
	if (m_addressSpace->blocks().empty() && m_retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM)) {
		m_addressSpace->addBlock(Retro::ramBase(m_core), m_retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM), m_retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM));
	}
}

vector<string> Emulator::buttons() const {
	return Retro::buttons(m_core);
}

vector<string> Emulator::keybinds() const {
	return Retro::keybinds(m_core);
}

bool Emulator::isHWRenderEnabled() const {
#ifdef ENABLE_HW_RENDER
	return m_hwRender.isEnabled();
#else
	return false;
#endif
}

#ifdef ENABLE_HW_RENDER
uintptr_t Emulator::cbGetCurrentFramebuffer() {
	return callbackEmulator()->m_hwRender.getCurrentFramebuffer();
}

retro_proc_address_t Emulator::cbGetProcAddress(const char* sym) {
	return callbackEmulator()->m_hwRender.getProcAddress(sym);
}
#endif
}
