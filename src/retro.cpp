#include <cmath>
#ifdef _WIN32
// pyconfig.h doesn't seem to like hypot, so we need to work around it.
namespace std {
template<typename T>
static inline T _hypot(T x, T y) {
	return hypot(x, y);
}
}
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "coreinfo.h"
#include "data.h"
#include "emulator.h"
#include "imageops.h"
#include "memory.h"
#include "search.h"
#include "script.h"
#include "movie.h"
#include "movie-bk2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

using std::string;
using namespace Retro;

static bool envFlagEnabled(const char* name) {
	const char* value = std::getenv(name);
	if (!value || !*value) {
		return false;
	}
	return std::strcmp(value, "0") != 0 &&
		std::strcmp(value, "false") != 0 &&
		std::strcmp(value, "False") != 0 &&
		std::strcmp(value, "FALSE") != 0;
}

struct AreaResizeBin {
	long y0 = 0;
	long y1 = 0;
	long x0 = 0;
	long x1 = 0;
	uint32_t count = 1;
};

struct AreaResizePlan {
	long dstH = 0;
	long dstW = 0;
	std::vector<AreaResizeBin> bins;
};

struct PyGameData;
struct PyRetroEmulator {
	Retro::Emulator m_re;
	int m_cheats = 0;
	PyRetroEmulator(const string& rom_path) {
		if (!m_re.loadRom(rom_path.c_str())) {
			throw std::runtime_error("Could not load ROM");
		}
		m_re.run(); // otherwise you get a segfault when you try to get screen for the first time
	}

	void step() {
		m_re.run();
	}

	void readRgbFrame(std::vector<uint8_t>& rgb) {
		long w = m_re.getImageWidth();
		long h = m_re.getImageHeight();
		rgb.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
		Image out(Image::Format::RGB888, rgb.data(), w, h, w);
		const void* img = m_re.getImageData();
		if (!img) {
			for (int i = 0; i < 180 && !img; ++i) {
				m_re.run();
				img = m_re.getImageData();
			}
		}
		if (!img) {
			throw std::runtime_error(
				"Core did not provide a CPU framebuffer. "
				"This usually means the core is using hardware rendering, which stable-retro can't capture yet."
			);
		}
		Image in;
		if (m_re.getImageDepth() == 16) {
			in = Image(Image::Format::RGB565, img, w, h, m_re.getImagePitch());
		} else if (m_re.getImageDepth() == 32) {
			in = Image(Image::Format::RGBX888, img, w, h, m_re.getImagePitch());
		} else {
			throw std::runtime_error("Unsupported image depth from core");
		}
		in.copyTo(&out);
	}

	void readRawFrame(std::vector<uint8_t>& raw) {
		long h = m_re.getImageHeight();
		const size_t pitch = static_cast<size_t>(m_re.getImagePitch());
		const void* img = m_re.getImageData();
		if (!img) {
			for (int i = 0; i < 180 && !img; ++i) {
				m_re.run();
				img = m_re.getImageData();
			}
		}
		if (!img) {
			throw std::runtime_error(
				"Core did not provide a CPU framebuffer. "
				"This usually means the core is using hardware rendering, which stable-retro can't capture yet."
			);
		}
		if (m_re.getImageDepth() != 16 && m_re.getImageDepth() != 32) {
			throw std::runtime_error("Unsupported image depth from core");
		}
		raw.resize(static_cast<size_t>(h) * pitch);
		std::memcpy(raw.data(), img, raw.size());
	}

	py::array_t<uint8_t> processRgbFrame(
		const std::vector<uint8_t>& rgb,
		py::object cropObj,
		py::object resizeObj,
		bool grayscale,
		const string& algorithm,
		py::object maskCropObj,
		int cropFill
	) {
		long w = m_re.getImageWidth();
		long h = m_re.getImageHeight();

		long top = 0;
		long bottom = 0;
		long left = 0;
		long right = 0;
		if (!cropObj.is_none()) {
			auto crop = py::tuple(cropObj);
			if (crop.size() != 4) {
				throw std::runtime_error("crop must be a (top, bottom, left, right) tuple");
			}
			top = py::int_(crop[0]);
			bottom = py::int_(crop[1]);
			left = py::int_(crop[2]);
			right = py::int_(crop[3]);
		}
		long maskTop = 0;
		long maskBottom = 0;
		long maskLeft = 0;
		long maskRight = 0;
		if (!maskCropObj.is_none()) {
			auto maskCrop = py::tuple(maskCropObj);
			if (maskCrop.size() != 4) {
				throw std::runtime_error("mask crop must be a (top, bottom, left, right) tuple");
			}
			maskTop = py::int_(maskCrop[0]);
			maskBottom = py::int_(maskCrop[1]);
			maskLeft = py::int_(maskCrop[2]);
			maskRight = py::int_(maskCrop[3]);
			if (maskTop < 0 || maskBottom < 0 || maskLeft < 0 || maskRight < 0) {
				throw std::runtime_error("mask crop values must be non-negative");
			}
		}
		if (cropFill < 0 || cropFill > 255) {
			throw std::runtime_error("crop_fill must be between 0 and 255");
		}
		if (top < 0 || bottom < 0 || left < 0 || right < 0 || top + bottom >= h || left + right >= w) {
			throw std::runtime_error("crop removes the entire observation");
		}
		long srcH = h - top - bottom;
		long srcW = w - left - right;
		long dstH = srcH;
		long dstW = srcW;
		if (!resizeObj.is_none()) {
			auto resize = py::tuple(resizeObj);
			if (resize.size() != 2) {
				throw std::runtime_error("resize must be a (height, width) tuple");
			}
			dstH = py::int_(resize[0]);
			dstW = py::int_(resize[1]);
			if (dstH <= 0 || dstW <= 0) {
				throw std::runtime_error("resize height and width must be positive");
			}
		}

		int channels = grayscale ? 1 : 3;
		py::array_t<uint8_t> arr({ { dstH, dstW, channels } });
		uint8_t* dst = arr.mutable_data();
		const uint8_t fill = static_cast<uint8_t>(cropFill);

		auto isMasked = [&](long y, long x) -> bool {
			return y < maskTop
				|| (maskBottom > 0 && y >= srcH - maskBottom)
				|| x < maskLeft
				|| (maskRight > 0 && x >= srcW - maskRight);
		};

		auto srcPixel = [&](long y, long x, int c) -> uint8_t {
			if (isMasked(y, x)) {
				return fill;
			}
			const size_t offset = (static_cast<size_t>(top + y) * static_cast<size_t>(w) + static_cast<size_t>(left + x)) * 3;
			return rgb[offset + c];
		};
		auto srcGray = [&](long y, long x) -> uint8_t {
			if (isMasked(y, x)) {
				return fill;
			}
			const size_t offset = (static_cast<size_t>(top + y) * static_cast<size_t>(w) + static_cast<size_t>(left + x)) * 3;
			const uint32_t r = rgb[offset];
			const uint32_t g = rgb[offset + 1];
			const uint32_t b = rgb[offset + 2];
			return static_cast<uint8_t>((r * 77 + g * 150 + b * 29 + 128) >> 8);
		};
		auto writePixel = [&](long y, long x, int c, uint8_t value) {
			dst[(static_cast<size_t>(y) * static_cast<size_t>(dstW) + static_cast<size_t>(x)) * channels + c] = value;
		};

		if (algorithm == "area") {
			if (dstH > srcH || dstW > srcW) {
				throw std::runtime_error("area resize only supports downscaling");
			}
			for (long dy = 0; dy < dstH; ++dy) {
				long y0 = (dy * srcH) / dstH;
				long y1 = std::max<long>(((dy + 1) * srcH) / dstH, y0 + 1);
				y1 = std::min(y1, srcH);
				for (long dx = 0; dx < dstW; ++dx) {
					long x0 = (dx * srcW) / dstW;
					long x1 = std::max<long>(((dx + 1) * srcW) / dstW, x0 + 1);
					x1 = std::min(x1, srcW);
					const uint32_t count = static_cast<uint32_t>((y1 - y0) * (x1 - x0));
					if (grayscale) {
						uint32_t sum = 0;
						for (long sy = y0; sy < y1; ++sy) {
							for (long sx = x0; sx < x1; ++sx) {
								sum += srcGray(sy, sx);
							}
						}
						writePixel(dy, dx, 0, static_cast<uint8_t>(sum / count));
					} else {
						for (int c = 0; c < 3; ++c) {
							uint32_t sum = 0;
							for (long sy = y0; sy < y1; ++sy) {
								for (long sx = x0; sx < x1; ++sx) {
									sum += srcPixel(sy, sx, c);
								}
							}
							writePixel(dy, dx, c, static_cast<uint8_t>(sum / count));
						}
					}
				}
			}
			return arr;
		}

		const bool bilinear = algorithm == "bilinear";
		for (long dy = 0; dy < dstH; ++dy) {
			const float fy = dstH == 1 ? 0.0f : static_cast<float>(dy) * static_cast<float>(srcH - 1) / static_cast<float>(dstH - 1);
			const long y0 = std::max<long>(0, std::min<long>(srcH - 1, static_cast<long>(std::floor(fy))));
			const long y1 = std::min<long>(y0 + 1, srcH - 1);
			const float wy = bilinear ? fy - static_cast<float>(y0) : 0.0f;
			const long sy = bilinear ? y0 : static_cast<long>(fy);
			for (long dx = 0; dx < dstW; ++dx) {
				const float fx = dstW == 1 ? 0.0f : static_cast<float>(dx) * static_cast<float>(srcW - 1) / static_cast<float>(dstW - 1);
				const long x0 = std::max<long>(0, std::min<long>(srcW - 1, static_cast<long>(std::floor(fx))));
				const long x1 = std::min<long>(x0 + 1, srcW - 1);
				const float wx = bilinear ? fx - static_cast<float>(x0) : 0.0f;
				const long sx = bilinear ? x0 : static_cast<long>(fx);
				if (!bilinear) {
					if (grayscale) {
						writePixel(dy, dx, 0, srcGray(sy, sx));
					} else {
						for (int c = 0; c < 3; ++c) {
							writePixel(dy, dx, c, srcPixel(sy, sx, c));
						}
					}
				} else if (grayscale) {
					const float topPix = static_cast<float>(srcGray(y0, x0)) * (1.0f - wx) + static_cast<float>(srcGray(y0, x1)) * wx;
					const float bottomPix = static_cast<float>(srcGray(y1, x0)) * (1.0f - wx) + static_cast<float>(srcGray(y1, x1)) * wx;
					writePixel(dy, dx, 0, static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, topPix * (1.0f - wy) + bottomPix * wy))));
				} else {
					for (int c = 0; c < 3; ++c) {
						const float topPix = static_cast<float>(srcPixel(y0, x0, c)) * (1.0f - wx) + static_cast<float>(srcPixel(y0, x1, c)) * wx;
						const float bottomPix = static_cast<float>(srcPixel(y1, x0, c)) * (1.0f - wx) + static_cast<float>(srcPixel(y1, x1, c)) * wx;
						writePixel(dy, dx, c, static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, topPix * (1.0f - wy) + bottomPix * wy))));
					}
				}
			}
		}
		return arr;
	}

	py::bytes getState() {
		size_t size = m_re.serializeSize();
		py::bytes bytes(NULL, size);
		m_re.serialize(PyBytes_AsString(bytes.ptr()), size);
		return bytes;
	}

	bool setState(py::bytes o) {
		return m_re.unserialize(PyBytes_AsString(o.ptr()), PyBytes_Size(o.ptr()));
	}

	py::array_t<uint8_t> getScreen() {
		long w = m_re.getImageWidth();
		long h = m_re.getImageHeight();
		py::array_t<uint8_t> arr({ { h, w, 3 } });
		uint8_t* data = arr.mutable_data();
		Image out(Image::Format::RGB888, data, w, h, w);
		const void* img = m_re.getImageData();
		if (!img) {
			// Some cores (notably N64) can take a number of frames before the first CPU framebuffer is produced.
			for (int i = 0; i < 180 && !img; ++i) {
				m_re.run();
				img = m_re.getImageData();
			}
		}
		if (!img) {
			throw std::runtime_error(
				"Core did not provide a CPU framebuffer. "
				"This usually means the core is using hardware rendering, which stable-retro can't capture yet. "
				"For N64/parallel_n64, try forcing a software renderer (parallel-n64-gfxplugin=angrylion)."
			);
		}
		Image in;
		if (m_re.getImageDepth() == 16) {
			in = Image(Image::Format::RGB565, img, w, h, m_re.getImagePitch());
		} else if (m_re.getImageDepth() == 32) {
			in = Image(Image::Format::RGBX888, img, w, h, m_re.getImagePitch());
		} else {
			throw std::runtime_error("Unsupported image depth from core");
		}
		in.copyTo(&out);
		return arr;
	}

	py::array_t<uint8_t> getProcessedScreen(
		py::object cropObj,
		py::object resizeObj,
		bool grayscale,
		const string& algorithm,
		py::object maskCropObj,
		int cropFill
	) {
		std::vector<uint8_t> rgb;
		readRgbFrame(rgb);
		return processRgbFrame(rgb, cropObj, resizeObj, grayscale, algorithm, maskCropObj, cropFill);
	}

	double getScreenRate() {
		return m_re.getFrameRate();
	}

	py::array_t<int16_t> getAudio() {
		py::array_t<int16_t> arr(py::array::ShapeContainer{ m_re.getAudioSamples(), 2 });
		int16_t* data = arr.mutable_data();
		memcpy(data, m_re.getAudioData(), m_re.getAudioSamples() * 4);
		return arr;
	}

	double getAudioRate() {
		return m_re.getAudioRate();
	}

	py::tuple getResolution() {
		return py::make_tuple(m_re.getImageWidth(), m_re.getImageHeight());
	}

	int getRotation() const {
		return m_re.getRotation();
	}

	void setButtonMask(py::array_t<uint8_t> mask, unsigned player) {
		if (mask.size() > N_BUTTONS) {
			throw std::runtime_error("mask.size() > N_BUTTONS");
		}
		if (player >= MAX_PLAYERS) {
			throw std::runtime_error("player >= MAX_PLAYERS");
		}
		for (int key = 0; key < mask.size(); ++key) {
			m_re.setKey(player, key, mask.data()[key]);
		}
	}

	void addCheat(const string& code) {
		m_re.setCheat(m_cheats, true, code.c_str());
		++m_cheats;
	}

	void clearCheats() {
		m_re.clearCheats();
		m_cheats = 0;
	}

	void configureData(PyGameData& data);
	py::tuple stepRepeatAndProcess(
		PyGameData& data,
		py::array_t<uint8_t> mask,
		int repeats,
		py::object cropObj,
		py::object resizeObj,
		bool grayscale,
		const string& algorithm,
		bool maxpoolLastTwo,
		py::object maskCropObj,
		int cropFill
	);
	static bool loadCoreInfo(const string& json) {
		return Retro::loadCoreInfo(json);
	}
};

struct PyMemoryView {
	Retro::AddressSpace& m_mem;
	PyMemoryView(Retro::AddressSpace& mem)
		: m_mem(mem) {
	}

	int64_t extract(size_t address, const string& type) {
		return m_mem[Variable{ type, address }];
	}

	void assign(size_t address, const string& type, int64_t value) {
		m_mem[Variable{ type, address }] = value;
	}

	void setitem(py::dict item, int64_t value) {
		return assign(py::int_(item["address"]), py::str(item["type"]), value);
	}

	int64_t getitem(py::dict item) {
		return extract(py::int_(item["address"]), py::str(item["type"]));
	}

	py::dict blocks() {
		py::dict obj;
		for (const auto& iter : m_mem.blocks()) {
			obj[py::int_(iter.first)] = py::bytes(static_cast<const char*>(iter.second.offset(0)), iter.second.size());
		}
		return obj;
	}
};

struct PySearch {
	Retro::Search* m_search;
	bool m_managed = true;
	PySearch(py::handle types) {
		if (!types.is_none()) {
			std::vector<Retro::DataType> dtypes;
			for (const auto& type : types) {
				dtypes.emplace_back(py::str(type));
			}
			m_search = new Retro::Search(dtypes);
		}
	}

	PySearch(Retro::Search* search) {
		m_search = search;
		m_managed = false;
	}

	~PySearch() {
		if (m_managed) {
			delete m_search;
		}
	}

	int numResults() const {
		return m_search->numResults();
	}

	bool hasUniqueResult() const {
		return m_search->hasUniqueResult();
	}

	py::dict uniqueResult() const {
		TypedSearchResult result = m_search->uniqueResult();
		py::dict obj;
		obj["address"] = result.address;
		obj["type"] = result.type.type;
		return obj;
	}

	py::list typedResults() const {
		std::map<SearchResult, std::unordered_set<DataType>> results;
		for (const auto& result : m_search->typedResults()) {
			results[static_cast<const SearchResult&>(result)].emplace(result.type);
		}
		py::list flattedResults;
		for (const auto& result : results) {
			py::list typeStrings;
			for (const auto& type : result.second) {
				typeStrings.append(py::str(type.type));
			}
			flattedResults.append(py::make_tuple(
				py::make_tuple(
					result.first.address,
					result.first.mult,
					result.first.div,
					result.first.bias),
				typeStrings));
		}
		return flattedResults;
	}
};

struct PyGameData {
	Retro::GameData m_data;
	Retro::Scenario m_scen{ m_data };

	bool load(py::handle data = py::none(), py::handle scen = py::none()) {
		ScriptContext::reset();

		bool success = true;
		if (!data.is_none()) {
			success = success && m_data.load(py::str(data));
		}
		if (!scen.is_none()) {
			success = success && m_scen.load(py::str(scen));
		}
		return success;
	}

	bool save(py::handle data = py::none(), py::handle scen = py::none()) {
		bool success = true;
		if (!data.is_none()) {
			success = success && m_data.save(py::str(data));
		}
		if (!scen.is_none()) {
			success = success && m_scen.save(py::str(scen));
		}
		return success;
	}

	void reset() {
		m_scen.restart();
		m_scen.reloadScripts();
	}

	uint16_t filterAction(uint16_t action) const {
		return m_scen.filterAction(action);
	}

	py::list validActions() const {
		py::list outer;
		for (const auto& action : m_scen.validActions()) {
			py::list inner;
			for (const auto& act : action.second) {
				inner.append(act);
			}
			outer.append(inner);
		}
		return outer;
	}

	void updateRam() {
		m_data.updateRam();
		m_scen.update();
	}

	static py::object variantToPython(Variant data) {
		switch (data.type()) {
		case Variant::Type::BOOL:
			return static_cast<py::bool_>(data);
		case Variant::Type::INT:
			return static_cast<py::int_>(static_cast<int64_t>(data));
		case Variant::Type::FLOAT:
			return static_cast<py::float_>(static_cast<double>(data));
		case Variant::Type::VOID:
			return py::none();
		}
		return py::none();
	}

	py::object lookupValue(py::str name) const {
		try {
			return variantToPython(m_data.lookupValue(name));
		} catch (std::invalid_argument e) {
			throw pybind11::key_error(e.what());
		}
	}

	py::object setValue(py::str name, py::object value) {
		if (py::isinstance<py::bool_>(value)) {
			m_data.setValue(name, Variant(static_cast<bool>(py::bool_(value))));
		}
		if (py::isinstance<py::int_>(value)) {
			m_data.setValue(name, Variant(static_cast<int64_t>(py::int_(value))));
		}
		if (py::isinstance<py::float_>(value)) {
			m_data.setValue(name, Variant(static_cast<double>(py::float_(value))));
		}
		if (value.is_none()) {
			m_data.setValue(name, Variant());
		}
		return value;
	}

	py::dict lookupAll() const {
		py::dict data;
		for (const auto& var : m_data.lookupAll()) {
			data[py::str(var.first)] = var.second;
		}
		return data;
	}

	py::dict getVariable(py::str name) const {
		py::dict obj;
		Retro::Variable var = m_data.getVariable(name);
		obj["address"] = var.address;
		obj["type"] = var.type.type;
		return obj;
	}

	void setVariable(py::str name, py::dict obj) {
		Retro::Variable var{ string(py::str(obj["type"])), py::int_(obj["address"]) };
		m_data.setVariable(name, var);
	}

	void removeVariable(py::str name) {
		m_data.removeVariable(name);
	}

	py::dict listVariables() {
		const auto& vars = m_data.listVariables();
		py::dict vdict;
		for (const auto& var : vars) {
			const auto& v = var.second;
			vdict[py::str(var.first)] = py::dict(py::arg("address") = v.address, py::arg("type") = v.type.type);
		}
		return vdict;
	}

	float currentReward(unsigned player = 0) const {
		return m_scen.currentReward(player);
	}

	float totalReward(unsigned player = 0) const {
		return m_scen.totalReward(player);
	}

	bool isDone() const {
		return m_scen.isDone();
	}

	py::tuple cropInfo(unsigned player = 0) {
		size_t x = 0;
		size_t y = 0;
		size_t width = 0;
		size_t height = 0;
		m_scen.getCrop(&x, &y, &width, &height, player);
		return py::make_tuple(x, y, width, height);
	}

	PyMemoryView memory() {
		return PyMemoryView(m_data.addressSpace());
	}

	void search(py::str name, int64_t value) {
		m_data.search(name, value);
	}

	void deltaSearch(py::str name, py::str op, int64_t ref) {
		m_data.deltaSearch(name, Retro::Scenario::op(op), ref);
	}

	PySearch getSearch(py::str name) {
		return m_data.getSearch(name);
	}

	void removeSearch(py::str name) {
		m_data.removeSearch(name);
	}

	py::dict listSearches() {
		const auto& names = m_data.listSearches();
		py::dict searches;
		for (const auto& name : names) {
			searches[py::str(name)] = PySearch(m_data.getSearch(name));
		}
		return searches;
	}
};

void PyRetroEmulator::configureData(PyGameData& data) {
	m_re.configureData(&data.m_data);
}

	class BatchThreadPool {
public:
	explicit BatchThreadPool(int numThreads) {
		for (int i = 0; i < numThreads; ++i) {
			m_threads.emplace_back([this]() { worker(); });
		}
	}

	~BatchThreadPool() {
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			m_stopping = true;
			++m_generation;
		}
		m_work.notify_all();
		for (auto& thread : m_threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}
	}

	void parallelFor(size_t count, std::function<void(size_t)> task) {
		if (count == 0) {
			return;
		}
		if (m_threads.empty() || count == 1) {
			for (size_t i = 0; i < count; ++i) {
				task(i);
			}
			return;
		}
		std::unique_lock<std::mutex> lock(m_mutex);
		m_task = std::move(task);
		m_total = count;
		m_next = 0;
		m_remainingWorkers = m_threads.size();
		const uint64_t generation = ++m_generation;
		m_work.notify_all();
		m_done.wait(lock, [&]() { return m_doneGeneration == generation; });
		m_task = nullptr;
	}

private:
	void worker() {
		uint64_t seenGeneration = 0;
		while (true) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_work.wait(lock, [&]() {
				return m_stopping || m_generation != seenGeneration;
			});
			if (m_stopping) {
				break;
			}
			seenGeneration = m_generation;
			auto* task = &m_task;
			while (true) {
				const size_t index = m_next++;
				if (index >= m_total) {
					break;
				}
				lock.unlock();
				(*task)(index);
				lock.lock();
			}
			if (--m_remainingWorkers == 0) {
				m_doneGeneration = seenGeneration;
				m_done.notify_one();
			}
		}
	}

	std::vector<std::thread> m_threads;
	std::mutex m_mutex;
	std::condition_variable m_work;
	std::condition_variable m_done;
	std::function<void(size_t)> m_task;
	size_t m_total = 0;
	size_t m_next = 0;
	size_t m_remainingWorkers = 0;
	uint64_t m_generation = 0;
	uint64_t m_doneGeneration = 0;
	bool m_stopping = false;
};

BatchThreadPool& batchThreadPool(int numThreads) {
	static std::mutex poolsMutex;
	static std::unordered_map<int, std::unique_ptr<BatchThreadPool>> pools;
	std::lock_guard<std::mutex> lock(poolsMutex);
	auto it = pools.find(numThreads);
	if (it == pools.end()) {
		it = pools.emplace(numThreads, std::make_unique<BatchThreadPool>(numThreads)).first;
	}
	return *it->second;
}

struct NativeCrop {
	long top = 0;
	long bottom = 0;
	long left = 0;
	long right = 0;
};

struct NativeResize {
	bool enabled = false;
	long height = 0;
	long width = 0;
};

NativeCrop parseNativeCrop(py::object cropObj) {
	NativeCrop crop;
	if (cropObj.is_none()) {
		return crop;
	}
	auto tuple = py::tuple(cropObj);
	if (tuple.size() != 4) {
		throw std::runtime_error("crop must be a (top, bottom, left, right) tuple");
	}
	crop.top = py::int_(tuple[0]);
	crop.bottom = py::int_(tuple[1]);
	crop.left = py::int_(tuple[2]);
	crop.right = py::int_(tuple[3]);
	if (crop.top < 0 || crop.bottom < 0 || crop.left < 0 || crop.right < 0) {
		throw std::runtime_error("crop values must be non-negative");
	}
	return crop;
}

NativeResize parseNativeResize(py::object resizeObj) {
	NativeResize resize;
	if (resizeObj.is_none()) {
		return resize;
	}
	auto tuple = py::tuple(resizeObj);
	if (tuple.size() != 2) {
		throw std::runtime_error("resize must be a (height, width) tuple");
	}
	resize.enabled = true;
	resize.height = py::int_(tuple[0]);
	resize.width = py::int_(tuple[1]);
	if (resize.height <= 0 || resize.width <= 0) {
		throw std::runtime_error("resize height and width must be positive");
	}
	return resize;
}

static inline bool cropMaskHasAny(const NativeCrop& mask) {
	return mask.top != 0 || mask.bottom != 0 || mask.left != 0 || mask.right != 0;
}

static inline bool cropMaskContains(const NativeCrop& mask, long height, long width, long y, long x) {
	return y < mask.top
		|| (mask.bottom > 0 && y >= height - mask.bottom)
		|| x < mask.left
		|| (mask.right > 0 && x >= width - mask.right);
}

void processRgbFrameToBuffer(
	const std::vector<uint8_t>& rgb,
	long rawW,
	long rawH,
	const NativeCrop& crop,
	const NativeCrop& maskCrop,
	const NativeResize& resize,
	bool grayscale,
	const string& algorithm,
	uint8_t cropFill,
	uint8_t* dst
) {
	if (crop.top + crop.bottom >= rawH || crop.left + crop.right >= rawW) {
		throw std::runtime_error("crop removes the entire observation");
	}
	const long srcH = rawH - crop.top - crop.bottom;
	const long srcW = rawW - crop.left - crop.right;
	const long dstH = resize.enabled ? resize.height : srcH;
	const long dstW = resize.enabled ? resize.width : srcW;
	const int channels = grayscale ? 1 : 3;

	auto srcPixel = [&](long y, long x, int c) -> uint8_t {
		if (cropMaskContains(maskCrop, srcH, srcW, y, x)) {
			return cropFill;
		}
		const size_t offset = (static_cast<size_t>(crop.top + y) * static_cast<size_t>(rawW) + static_cast<size_t>(crop.left + x)) * 3;
		return rgb[offset + c];
	};
	auto srcGray = [&](long y, long x) -> uint8_t {
		if (cropMaskContains(maskCrop, srcH, srcW, y, x)) {
			return cropFill;
		}
		const size_t offset = (static_cast<size_t>(crop.top + y) * static_cast<size_t>(rawW) + static_cast<size_t>(crop.left + x)) * 3;
		const uint32_t r = rgb[offset];
		const uint32_t g = rgb[offset + 1];
		const uint32_t b = rgb[offset + 2];
		return static_cast<uint8_t>((r * 77 + g * 150 + b * 29 + 128) >> 8);
	};
	auto writePixel = [&](long y, long x, int c, uint8_t value) {
		dst[(static_cast<size_t>(y) * static_cast<size_t>(dstW) + static_cast<size_t>(x)) * channels + c] = value;
	};

	if (algorithm == "area") {
		if (dstH > srcH || dstW > srcW) {
			throw std::runtime_error("area resize only supports downscaling");
		}
		for (long dy = 0; dy < dstH; ++dy) {
			long y0 = (dy * srcH) / dstH;
			long y1 = std::max<long>(((dy + 1) * srcH) / dstH, y0 + 1);
			y1 = std::min(y1, srcH);
			for (long dx = 0; dx < dstW; ++dx) {
				long x0 = (dx * srcW) / dstW;
				long x1 = std::max<long>(((dx + 1) * srcW) / dstW, x0 + 1);
				x1 = std::min(x1, srcW);
				const uint32_t count = static_cast<uint32_t>((y1 - y0) * (x1 - x0));
				if (grayscale) {
					uint32_t sum = 0;
					for (long sy = y0; sy < y1; ++sy) {
						for (long sx = x0; sx < x1; ++sx) {
							sum += srcGray(sy, sx);
						}
					}
					writePixel(dy, dx, 0, static_cast<uint8_t>(sum / count));
				} else {
					for (int c = 0; c < 3; ++c) {
						uint32_t sum = 0;
						for (long sy = y0; sy < y1; ++sy) {
							for (long sx = x0; sx < x1; ++sx) {
								sum += srcPixel(sy, sx, c);
							}
						}
						writePixel(dy, dx, c, static_cast<uint8_t>(sum / count));
					}
				}
			}
		}
		return;
	}

	const bool bilinear = algorithm == "bilinear";
	for (long dy = 0; dy < dstH; ++dy) {
		const float fy = dstH == 1 ? 0.0f : static_cast<float>(dy) * static_cast<float>(srcH - 1) / static_cast<float>(dstH - 1);
		const long y0 = std::max<long>(0, std::min<long>(srcH - 1, static_cast<long>(std::floor(fy))));
		const long y1 = std::min<long>(y0 + 1, srcH - 1);
		const float wy = bilinear ? fy - static_cast<float>(y0) : 0.0f;
		const long sy = bilinear ? y0 : static_cast<long>(fy);
		for (long dx = 0; dx < dstW; ++dx) {
			const float fx = dstW == 1 ? 0.0f : static_cast<float>(dx) * static_cast<float>(srcW - 1) / static_cast<float>(dstW - 1);
			const long x0 = std::max<long>(0, std::min<long>(srcW - 1, static_cast<long>(std::floor(fx))));
			const long x1 = std::min<long>(x0 + 1, srcW - 1);
			const float wx = bilinear ? fx - static_cast<float>(x0) : 0.0f;
			const long sx = bilinear ? x0 : static_cast<long>(fx);
			if (!bilinear) {
				if (grayscale) {
					writePixel(dy, dx, 0, srcGray(sy, sx));
				} else {
					for (int c = 0; c < 3; ++c) {
						writePixel(dy, dx, c, srcPixel(sy, sx, c));
					}
				}
			} else if (grayscale) {
				const float topPix = static_cast<float>(srcGray(y0, x0)) * (1.0f - wx) + static_cast<float>(srcGray(y0, x1)) * wx;
				const float bottomPix = static_cast<float>(srcGray(y1, x0)) * (1.0f - wx) + static_cast<float>(srcGray(y1, x1)) * wx;
				writePixel(dy, dx, 0, static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, topPix * (1.0f - wy) + bottomPix * wy))));
			} else {
				for (int c = 0; c < 3; ++c) {
					const float topPix = static_cast<float>(srcPixel(y0, x0, c)) * (1.0f - wx) + static_cast<float>(srcPixel(y0, x1, c)) * wx;
					const float bottomPix = static_cast<float>(srcPixel(y1, x0, c)) * (1.0f - wx) + static_cast<float>(srcPixel(y1, x1, c)) * wx;
					writePixel(dy, dx, c, static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, topPix * (1.0f - wy) + bottomPix * wy))));
				}
			}
		}
	}
}

static inline void nativePixelChannels(
	const uint8_t* raw,
	size_t pitch,
	int depth,
	long y,
	long x,
	uint8_t& r,
	uint8_t& g,
	uint8_t& b
) {
	const uint8_t* row = raw + static_cast<size_t>(y) * pitch;
	if (depth == 16) {
		uint16_t rgb;
		std::memcpy(&rgb, row + static_cast<size_t>(x) * 2, sizeof(rgb));
		r = static_cast<uint8_t>((rgb & 0xF800) >> 8);
		g = static_cast<uint8_t>((rgb & 0x07E0) >> 3);
		b = static_cast<uint8_t>((rgb & 0x001F) << 3);
		return;
	}
	uint32_t xrgb;
	std::memcpy(&xrgb, row + static_cast<size_t>(x) * 4, sizeof(xrgb));
	r = static_cast<uint8_t>(xrgb);
	g = static_cast<uint8_t>(xrgb >> 8);
	b = static_cast<uint8_t>(xrgb >> 16);
}

static inline uint8_t nativeGray(
	const uint8_t* raw,
	const uint8_t* maxRaw,
	size_t pitch,
	int depth,
	long y,
	long x
) {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	nativePixelChannels(raw, pitch, depth, y, x, r, g, b);
	if (maxRaw) {
		uint8_t r2;
		uint8_t g2;
		uint8_t b2;
		nativePixelChannels(maxRaw, pitch, depth, y, x, r2, g2, b2);
		r = std::max(r, r2);
		g = std::max(g, g2);
		b = std::max(b, b2);
	}
	return static_cast<uint8_t>((static_cast<uint32_t>(r) * 77 + static_cast<uint32_t>(g) * 150 + static_cast<uint32_t>(b) * 29 + 128) >> 8);
}

static inline uint8_t rgb565Gray(uint16_t rgb) {
	const uint32_t r = (rgb & 0xF800) >> 8;
	const uint32_t g = (rgb & 0x07E0) >> 3;
	const uint32_t b = (rgb & 0x001F) << 3;
	return static_cast<uint8_t>((r * 77 + g * 150 + b * 29 + 128) >> 8);
}

static inline void rgb565ToRgb888(uint16_t rgb, uint8_t& r, uint8_t& g, uint8_t& b) {
	r = static_cast<uint8_t>(((rgb >> 11) & 0x1F) * 255 / 31);
	g = static_cast<uint8_t>(((rgb >> 5) & 0x3F) * 255 / 63);
	b = static_cast<uint8_t>((rgb & 0x1F) * 255 / 31);
}

static inline uint8_t rgb565MaxGray(uint16_t rgb, uint16_t maxRgb) {
	const uint32_t r = std::max<uint32_t>((rgb & 0xF800) >> 8, (maxRgb & 0xF800) >> 8);
	const uint32_t g = std::max<uint32_t>((rgb & 0x07E0) >> 3, (maxRgb & 0x07E0) >> 3);
	const uint32_t b = std::max<uint32_t>((rgb & 0x001F) << 3, (maxRgb & 0x001F) << 3);
	return static_cast<uint8_t>((r * 77 + g * 150 + b * 29 + 128) >> 8);
}

static inline uint16_t indexedRgb565(const IndexedVideoFrame& frame, uint8_t pixel) {
	if (frame.rawPalette) {
		return static_cast<uint16_t>(frame.palette[pixel & 0x3F] | frame.deemp);
	}
	return frame.palette[pixel];
}

struct IndexedPaletteCache {
	bool valid = false;
	bool rawPalette = false;
	int deemp = 0;
	std::array<uint16_t, 256> palette{};
	std::array<uint8_t, 256> gray{};
	std::array<uint8_t, 65536> maxGray{};
};

void updateIndexedPaletteCache(const IndexedVideoFrame& frame, IndexedPaletteCache& cache) {
	bool changed = !cache.valid || cache.rawPalette != frame.rawPalette || cache.deemp != frame.deemp;
	if (!changed) {
		for (size_t i = 0; i < cache.palette.size(); ++i) {
			if (cache.palette[i] != frame.palette[i]) {
				changed = true;
				break;
			}
		}
	}
	if (!changed) {
		return;
	}
	cache.valid = true;
	cache.rawPalette = frame.rawPalette;
	cache.deemp = frame.deemp;
	for (size_t i = 0; i < cache.palette.size(); ++i) {
		const uint16_t rgb = frame.rawPalette
			? static_cast<uint16_t>(frame.palette[i & 0x3F] | frame.deemp)
			: frame.palette[i];
		cache.palette[i] = frame.palette[i];
		cache.gray[i] = rgb565Gray(rgb);
	}
	for (size_t curr = 0; curr < 256; ++curr) {
		const uint16_t currRgb = frame.rawPalette
			? static_cast<uint16_t>(frame.palette[curr & 0x3F] | frame.deemp)
			: frame.palette[curr];
		uint8_t* row = cache.maxGray.data() + curr * 256;
		for (size_t prev = 0; prev < 256; ++prev) {
			const uint16_t prevRgb = frame.rawPalette
				? static_cast<uint16_t>(frame.palette[prev & 0x3F] | frame.deemp)
				: frame.palette[prev];
			row[prev] = rgb565MaxGray(currRgb, prevRgb);
		}
	}
}

static inline uint8_t indexedGray(
	const IndexedVideoFrame& frame,
	const uint8_t* maxRaw,
	long y,
	long x
) {
	const uint8_t pixel = frame.data[static_cast<size_t>(y) * frame.pitch + static_cast<size_t>(x)];
	if (maxRaw) {
		const uint8_t maxPixel = maxRaw[static_cast<size_t>(y) * frame.pitch + static_cast<size_t>(x)];
		const uint16_t curr = indexedRgb565(frame, pixel);
		const uint16_t prev = indexedRgb565(frame, maxPixel);
		return rgb565MaxGray(curr, prev);
	}
	return rgb565Gray(indexedRgb565(frame, pixel));
}

static inline uint8_t xrgb8888Gray(uint32_t xrgb) {
	const uint32_t r = xrgb & 0xFF;
	const uint32_t g = (xrgb >> 8) & 0xFF;
	const uint32_t b = (xrgb >> 16) & 0xFF;
	return static_cast<uint8_t>((r * 77 + g * 150 + b * 29 + 128) >> 8);
}

static inline uint8_t xrgb8888MaxGray(uint32_t xrgb, uint32_t maxXrgb) {
	const uint32_t r = std::max<uint32_t>(xrgb & 0xFF, maxXrgb & 0xFF);
	const uint32_t g = std::max<uint32_t>((xrgb >> 8) & 0xFF, (maxXrgb >> 8) & 0xFF);
	const uint32_t b = std::max<uint32_t>((xrgb >> 16) & 0xFF, (maxXrgb >> 16) & 0xFF);
	return static_cast<uint8_t>((r * 77 + g * 150 + b * 29 + 128) >> 8);
}

AreaResizePlan buildAreaResizePlan(long rawW, long rawH, const NativeCrop& crop, const NativeResize& resize) {
	if (crop.top + crop.bottom >= rawH || crop.left + crop.right >= rawW) {
		throw std::runtime_error("crop removes the entire observation");
	}
	const long srcH = rawH - crop.top - crop.bottom;
	const long srcW = rawW - crop.left - crop.right;
	const long dstH = resize.enabled ? resize.height : srcH;
	const long dstW = resize.enabled ? resize.width : srcW;
	if (dstH > srcH || dstW > srcW) {
		throw std::runtime_error("area resize only supports downscaling");
	}
	AreaResizePlan plan;
	plan.dstH = dstH;
	plan.dstW = dstW;
	plan.bins.reserve(static_cast<size_t>(dstH) * static_cast<size_t>(dstW));
	for (long dy = 0; dy < dstH; ++dy) {
		long y0 = (dy * srcH) / dstH;
		long y1 = std::max<long>(((dy + 1) * srcH) / dstH, y0 + 1);
		y1 = std::min(y1, srcH);
		for (long dx = 0; dx < dstW; ++dx) {
			long x0 = (dx * srcW) / dstW;
			long x1 = std::max<long>(((dx + 1) * srcW) / dstW, x0 + 1);
			x1 = std::min(x1, srcW);
			AreaResizeBin bin;
			bin.y0 = crop.top + y0;
			bin.y1 = crop.top + y1;
			bin.x0 = crop.left + x0;
			bin.x1 = crop.left + x1;
			bin.count = static_cast<uint32_t>((y1 - y0) * (x1 - x0));
			plan.bins.push_back(bin);
		}
	}
	return plan;
}

void processNativeGrayscaleAreaPlanToBuffer(
	const uint8_t* raw,
	const uint8_t* maxRaw,
	size_t pitch,
	int depth,
	const AreaResizePlan& plan,
	uint8_t* dst
) {
	if (depth == 16) {
		for (size_t i = 0; i < plan.bins.size(); ++i) {
			const AreaResizeBin& bin = plan.bins[i];
			uint32_t sum = 0;
			for (long sy = bin.y0; sy < bin.y1; ++sy) {
				const uint8_t* row = raw + static_cast<size_t>(sy) * pitch;
				const uint8_t* maxRow = maxRaw ? maxRaw + static_cast<size_t>(sy) * pitch : nullptr;
				for (long sx = bin.x0; sx < bin.x1; ++sx) {
					uint16_t rgb;
					std::memcpy(&rgb, row + static_cast<size_t>(sx) * 2, sizeof(rgb));
					if (maxRow) {
						uint16_t maxRgb;
						std::memcpy(&maxRgb, maxRow + static_cast<size_t>(sx) * 2, sizeof(maxRgb));
						sum += rgb565MaxGray(rgb, maxRgb);
					} else {
						sum += rgb565Gray(rgb);
					}
				}
			}
			dst[i] = static_cast<uint8_t>(sum / bin.count);
		}
		return;
	}
	if (depth == 32) {
		for (size_t i = 0; i < plan.bins.size(); ++i) {
			const AreaResizeBin& bin = plan.bins[i];
			uint32_t sum = 0;
			for (long sy = bin.y0; sy < bin.y1; ++sy) {
				const uint8_t* row = raw + static_cast<size_t>(sy) * pitch;
				const uint8_t* maxRow = maxRaw ? maxRaw + static_cast<size_t>(sy) * pitch : nullptr;
				for (long sx = bin.x0; sx < bin.x1; ++sx) {
					uint32_t xrgb;
					std::memcpy(&xrgb, row + static_cast<size_t>(sx) * 4, sizeof(xrgb));
					if (maxRow) {
						uint32_t maxXrgb;
						std::memcpy(&maxXrgb, maxRow + static_cast<size_t>(sx) * 4, sizeof(maxXrgb));
						sum += xrgb8888MaxGray(xrgb, maxXrgb);
					} else {
						sum += xrgb8888Gray(xrgb);
					}
				}
			}
			dst[i] = static_cast<uint8_t>(sum / bin.count);
		}
		return;
	}
	throw std::runtime_error("Unsupported image depth from core");
}

void processIndexedGrayscaleAreaPlanToBuffer(
	const IndexedVideoFrame& frame,
	const uint8_t* maxRaw,
	IndexedPaletteCache* cache,
	const AreaResizePlan& plan,
	uint8_t* dst
) {
	if (cache) {
		updateIndexedPaletteCache(frame, *cache);
		const uint8_t* gray = cache->gray.data();
		const uint8_t* maxGray = cache->maxGray.data();
		for (size_t i = 0; i < plan.bins.size(); ++i) {
			const AreaResizeBin& bin = plan.bins[i];
			uint32_t sum = 0;
			for (long sy = bin.y0; sy < bin.y1; ++sy) {
				const uint8_t* row = frame.data + static_cast<size_t>(sy) * frame.pitch;
				const uint8_t* maxRow = maxRaw ? maxRaw + static_cast<size_t>(sy) * frame.pitch : nullptr;
				for (long sx = bin.x0; sx < bin.x1; ++sx) {
					const uint8_t pixel = row[sx];
					sum += maxRow ? maxGray[(static_cast<size_t>(pixel) << 8) | maxRow[sx]] : gray[pixel];
				}
			}
			dst[i] = static_cast<uint8_t>(sum / bin.count);
		}
		return;
	}
	for (size_t i = 0; i < plan.bins.size(); ++i) {
		const AreaResizeBin& bin = plan.bins[i];
		uint32_t sum = 0;
		for (long sy = bin.y0; sy < bin.y1; ++sy) {
			for (long sx = bin.x0; sx < bin.x1; ++sx) {
				sum += indexedGray(frame, maxRaw, sy, sx);
			}
		}
		dst[i] = static_cast<uint8_t>(sum / bin.count);
	}
}

void processNativeGrayscaleFrameToBuffer(
	const uint8_t* raw,
	const uint8_t* maxRaw,
	long rawW,
	long rawH,
	size_t pitch,
	int depth,
	const NativeCrop& crop,
	const NativeCrop& maskCrop,
	const NativeResize& resize,
	const string& algorithm,
	uint8_t cropFill,
	uint8_t* dst
) {
	if (crop.top + crop.bottom >= rawH || crop.left + crop.right >= rawW) {
		throw std::runtime_error("crop removes the entire observation");
	}
	if (depth != 16 && depth != 32) {
		throw std::runtime_error("Unsupported image depth from core");
	}
	const long srcH = rawH - crop.top - crop.bottom;
	const long srcW = rawW - crop.left - crop.right;
	const long dstH = resize.enabled ? resize.height : srcH;
	const long dstW = resize.enabled ? resize.width : srcW;

	auto srcGray = [&](long y, long x) -> uint8_t {
		if (cropMaskContains(maskCrop, srcH, srcW, y, x)) {
			return cropFill;
		}
		return nativeGray(raw, maxRaw, pitch, depth, crop.top + y, crop.left + x);
	};
	auto writePixel = [&](long y, long x, uint8_t value) {
		dst[static_cast<size_t>(y) * static_cast<size_t>(dstW) + static_cast<size_t>(x)] = value;
	};

	if (algorithm == "area") {
		if (dstH > srcH || dstW > srcW) {
			throw std::runtime_error("area resize only supports downscaling");
		}
		for (long dy = 0; dy < dstH; ++dy) {
			long y0 = (dy * srcH) / dstH;
			long y1 = std::max<long>(((dy + 1) * srcH) / dstH, y0 + 1);
			y1 = std::min(y1, srcH);
			for (long dx = 0; dx < dstW; ++dx) {
				long x0 = (dx * srcW) / dstW;
				long x1 = std::max<long>(((dx + 1) * srcW) / dstW, x0 + 1);
				x1 = std::min(x1, srcW);
				uint32_t sum = 0;
				for (long sy = y0; sy < y1; ++sy) {
					for (long sx = x0; sx < x1; ++sx) {
						sum += srcGray(sy, sx);
					}
				}
				const uint32_t count = static_cast<uint32_t>((y1 - y0) * (x1 - x0));
				writePixel(dy, dx, static_cast<uint8_t>(sum / count));
			}
		}
		return;
	}

	const bool bilinear = algorithm == "bilinear";
	for (long dy = 0; dy < dstH; ++dy) {
		const float fy = dstH == 1 ? 0.0f : static_cast<float>(dy) * static_cast<float>(srcH - 1) / static_cast<float>(dstH - 1);
		const long y0 = std::max<long>(0, std::min<long>(srcH - 1, static_cast<long>(std::floor(fy))));
		const long y1 = std::min<long>(y0 + 1, srcH - 1);
		const float wy = bilinear ? fy - static_cast<float>(y0) : 0.0f;
		const long sy = bilinear ? y0 : static_cast<long>(fy);
		for (long dx = 0; dx < dstW; ++dx) {
			const float fx = dstW == 1 ? 0.0f : static_cast<float>(dx) * static_cast<float>(srcW - 1) / static_cast<float>(dstW - 1);
			const long x0 = std::max<long>(0, std::min<long>(srcW - 1, static_cast<long>(std::floor(fx))));
			const long x1 = std::min<long>(x0 + 1, srcW - 1);
			const float wx = bilinear ? fx - static_cast<float>(x0) : 0.0f;
			const long sx = bilinear ? x0 : static_cast<long>(fx);
			if (!bilinear) {
				writePixel(dy, dx, srcGray(sy, sx));
			} else {
				const float topPix = static_cast<float>(srcGray(y0, x0)) * (1.0f - wx) + static_cast<float>(srcGray(y0, x1)) * wx;
				const float bottomPix = static_cast<float>(srcGray(y1, x0)) * (1.0f - wx) + static_cast<float>(srcGray(y1, x1)) * wx;
				writePixel(dy, dx, static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, topPix * (1.0f - wy) + bottomPix * wy))));
			}
		}
	}
}

class PyRetroVecEnv {
public:
	PyRetroVecEnv(
		size_t numEnvs,
		const string& romPath,
		const string& dataPath,
		const string& scenarioPath,
		py::object initialStateObj,
		int numButtons,
		int frameSkip,
		int frameStack,
		py::object cropObj,
		py::object resizeObj,
		bool grayscale,
		const string& algorithm,
		bool maxpoolLastTwo,
		int noopResetMax,
		double stickyActionProb,
		bool filterActions,
		bool rewardClip,
		float rewardClipLow,
		float rewardClipHigh,
		int numThreads,
		const string& infoFilterMode,
		bool unsafeView,
		const string& obsLayout,
		py::object infoFilterKeysObj,
		py::object doneOnRulesObj = py::none(),
		py::object initialStateLabelsObj = py::none(),
		py::object initialStateWeightsObj = py::none(),
		py::object maskCropObj = py::none(),
		int cropFill = 0
	)
		: m_numButtons(numButtons)
		, m_frameSkip(frameSkip)
		, m_frameStack(frameStack)
		, m_crop(parseNativeCrop(cropObj))
		, m_cropMask(parseNativeCrop(maskCropObj))
		, m_resize(parseNativeResize(resizeObj))
		, m_cropFill(static_cast<uint8_t>(cropFill))
		, m_grayscale(grayscale)
		, m_algorithm(algorithm)
		, m_maxpoolLastTwo(maxpoolLastTwo)
		, m_noopResetMax(noopResetMax)
		, m_stickyActionProb(stickyActionProb)
		, m_filterActions(filterActions)
		, m_rewardClip(rewardClip)
		, m_rewardClipLow(rewardClipLow)
		, m_rewardClipHigh(rewardClipHigh)
		, m_fullInfo(infoFilterMode == "all")
		, m_noInfo(infoFilterMode == "none")
		, m_unsafeZeroCopy(unsafeView)
		, m_numThreads(numThreads) {
		if (numEnvs == 0) {
			throw std::runtime_error("num_envs must be positive");
		}
		if (numButtons <= 0 || numButtons > N_BUTTONS) {
			throw std::runtime_error("num_buttons must be between 1 and 16");
		}
		if (frameSkip <= 0) {
			throw std::runtime_error("frame_skip must be positive");
		}
		if (frameStack <= 0) {
			throw std::runtime_error("frame_stack must be positive");
		}
		if (noopResetMax < 0) {
			throw std::runtime_error("noop_reset_max must be non-negative");
		}
		if (stickyActionProb < 0.0 || stickyActionProb > 1.0) {
			throw std::runtime_error("sticky_action_prob must be between 0.0 and 1.0");
		}
		if (cropFill < 0 || cropFill > 255) {
			throw std::runtime_error("crop_fill must be between 0 and 255");
		}
		if (algorithm != "nearest" && algorithm != "bilinear" && algorithm != "area") {
			throw std::runtime_error("algorithm must be nearest, bilinear, or area");
		}
		if (infoFilterMode != "terminal" && infoFilterMode != "all" && infoFilterMode != "none") {
			throw std::runtime_error("info_filter mode must be terminal, all, or none");
		}
		if (obsLayout == "hwc") {
			m_channelsFirst = false;
		} else if (obsLayout == "chw") {
			m_channelsFirst = true;
		} else {
			throw std::runtime_error("obs_layout must be hwc or chw");
		}
		parseDoneOnInfoRules(doneOnRulesObj);
		if (!infoFilterKeysObj.is_none()) {
			py::sequence infoKeys = py::reinterpret_borrow<py::sequence>(infoFilterKeysObj);
			m_infoKeys.reserve(static_cast<size_t>(infoKeys.size()));
			for (py::handle key : infoKeys) {
				if (!PyUnicode_Check(key.ptr())) {
					throw std::runtime_error("info_filter keys must contain only strings");
				}
				m_infoKeys.push_back(py::str(key));
			}
		}
		m_renderSkipEnabled = !envFlagEnabled("STABLE_RETRO_DISABLE_RENDER_SKIP");
		parseInitialStates(initialStateObj, initialStateLabelsObj, initialStateWeightsObj, numEnvs);
		m_numThreads = std::max(1, std::min<int>(m_numThreads <= 0 ? static_cast<int>(numEnvs) : m_numThreads, static_cast<int>(numEnvs)));
		const std::string& constructionInitialState = initialStateForConstruction();
		m_slots.reserve(numEnvs);
		for (size_t i = 0; i < numEnvs; ++i) {
			auto slot = std::make_unique<Slot>(romPath, dataPath, scenarioPath, constructionInitialState, i);
			if (m_grayscale && m_algorithm == "area") {
				slot->usesIndexedVideo = slot->emulator->m_re.setIndexedVideoEnabled(true);
			}
			if (i == 0) {
				const auto variables = slot->data.m_data.listVariables();
				resolveDoneOnInfoVariables(variables);
				if (!m_infoKeys.empty()) {
					for (const std::string& key : m_infoKeys) {
						auto variable = variables.find(key);
						if (variable == variables.end()) {
							throw std::runtime_error("unknown info key: " + key);
						}
						m_infoVariables.emplace_back(key, variable->second);
					}
				}
				const long rawW = slot->emulator->m_re.getImageWidth();
				const long rawH = slot->emulator->m_re.getImageHeight();
				if (m_crop.top + m_crop.bottom >= rawH || m_crop.left + m_crop.right >= rawW) {
					throw std::runtime_error("crop removes the entire observation");
				}
				const long srcH = rawH - m_crop.top - m_crop.bottom;
				const long srcW = rawW - m_crop.left - m_crop.right;
				m_obsHeight = m_resize.enabled ? m_resize.height : srcH;
				m_obsWidth = m_resize.enabled ? m_resize.width : srcW;
				m_obsChannels = m_grayscale ? 1 : 3;
				m_singleObsSize = static_cast<size_t>(m_obsHeight) * static_cast<size_t>(m_obsWidth) * static_cast<size_t>(m_obsChannels);
				m_stackedChannels = m_obsChannels * m_frameStack;
				m_stackedObsSize = static_cast<size_t>(m_obsHeight) * static_cast<size_t>(m_obsWidth) * static_cast<size_t>(m_stackedChannels);
				if (m_grayscale && m_algorithm == "area" && !cropMaskHasAny(m_cropMask)) {
					m_grayscaleAreaPlan = buildAreaResizePlan(rawW, rawH, m_crop, m_resize);
					m_useGrayscaleAreaPlan = true;
				}
			}
			slot->frameStack.resize(static_cast<size_t>(m_frameStack) * m_singleObsSize);
			slot->action.resize(static_cast<size_t>(m_numButtons));
			slot->lastMask.assign(static_cast<size_t>(m_numButtons), 0);
			slot->doneOnInfoBaselines.resize(m_doneOnInfoRules.size());
			m_slots.emplace_back(std::move(slot));
		}
		const auto obsShape = observationBatchShape();
		m_obsArrays[0] = py::array_t<uint8_t>(obsShape);
		m_obsArrays[1] = m_unsafeZeroCopy ? m_obsArrays[0] : py::array_t<uint8_t>(obsShape);
		py::array::ShapeContainer vecShape{ static_cast<py::ssize_t>(m_slots.size()) };
		m_rewardArray = py::array_t<float>(vecShape);
		m_doneArray = py::array_t<bool>(vecShape);
		m_activeInitialStateIndices.assign(m_slots.size(), -1);
		for (size_t i = 0; i < m_slots.size(); ++i) {
			m_emptyInfos.append(py::dict());
		}
		m_errors.resize(m_slots.size());
		m_terminalObservations.resize(m_slots.size());
	}

	enum class InitialStateMode {
		None,
		Single,
		Slot,
		Sample
	};

	struct InitialStateChoice {
		std::string state;
		std::string label;
		double weight = 1.0;
		int32_t nameIndex = -1;
	};

	int32_t internInitialStateName(const std::string& label) {
		auto it = std::find(m_initialStateNames.begin(), m_initialStateNames.end(), label);
		if (it != m_initialStateNames.end()) {
			return static_cast<int32_t>(std::distance(m_initialStateNames.begin(), it));
		}
		m_initialStateNames.push_back(label);
		return static_cast<int32_t>(m_initialStateNames.size() - 1);
	}

	std::vector<std::string> parseInitialStateLabels(py::object labelsObj, size_t stateCount) {
		std::vector<std::string> labels(stateCount);
		if (labelsObj.is_none()) {
			return labels;
		}
		py::sequence labelsSeq = py::reinterpret_borrow<py::sequence>(labelsObj);
		if (static_cast<size_t>(labelsSeq.size()) != stateCount) {
			throw std::runtime_error("initial_state_labels length must match initial_state length");
		}
		for (size_t i = 0; i < stateCount; ++i) {
			labels[i] = py::str(labelsSeq[static_cast<py::ssize_t>(i)]);
		}
		return labels;
	}

	void parseInitialStates(py::object initialStateObj, py::object labelsObj, py::object weightsObj, size_t numEnvs) {
		m_initialStates.clear();
		m_initialStateCumulative.clear();
		m_initialStateMode = InitialStateMode::None;
		m_reportInitialState = false;
		if (initialStateObj.is_none()) {
			if (!labelsObj.is_none() || !weightsObj.is_none()) {
				throw std::runtime_error("initial_state labels/weights require initial_state");
			}
			m_initialStateMode = InitialStateMode::None;
			return;
		}
		if (PyBytes_Check(initialStateObj.ptr())) {
			std::string state = py::bytes(initialStateObj);
			std::vector<std::string> labels = parseInitialStateLabels(labelsObj, 1);
			const int32_t nameIndex = internInitialStateName(labels[0]);
			m_initialStates.push_back({state, labels[0], 1.0, nameIndex});
			m_initialStateMode = InitialStateMode::Single;
			return;
		}

		py::sequence states = py::reinterpret_borrow<py::sequence>(initialStateObj);
		const size_t stateCount = static_cast<size_t>(states.size());
		if (stateCount == 0) {
			throw std::runtime_error("initial_state sequence must not be empty");
		}

		std::vector<std::string> labels = parseInitialStateLabels(labelsObj, stateCount);
		if (!labelsObj.is_none()) {
			m_reportInitialState = true;
		}

		std::vector<double> weights(stateCount, 1.0);
		if (!weightsObj.is_none()) {
			py::sequence weightsSeq = py::reinterpret_borrow<py::sequence>(weightsObj);
			if (static_cast<size_t>(weightsSeq.size()) != stateCount) {
				throw std::runtime_error("initial_state_weights length must match initial_state length");
			}
			double cumulative = 0.0;
			for (size_t i = 0; i < stateCount; ++i) {
				const double weight = py::float_(weightsSeq[static_cast<py::ssize_t>(i)]).cast<double>();
				if (!std::isfinite(weight) || weight < 0.0) {
					throw std::runtime_error("initial_state_weights must contain non-negative finite numbers");
				}
				weights[i] = weight;
				cumulative += weight;
				m_initialStateCumulative.push_back(cumulative);
			}
			if (!std::isfinite(cumulative) || cumulative <= 0.0) {
				throw std::runtime_error("initial_state_weights must sum to a positive number");
			}
			m_initialStateMode = InitialStateMode::Sample;
		} else {
			if (stateCount != numEnvs) {
				throw std::runtime_error("initial_state sequence length must match num_envs when initial_state_weights is not provided");
			}
			m_initialStateMode = InitialStateMode::Slot;
		}

		m_initialStates.reserve(stateCount);
		for (size_t i = 0; i < stateCount; ++i) {
			std::string state = py::bytes(states[static_cast<py::ssize_t>(i)]);
			if (state.empty()) {
				throw std::runtime_error("initial_state entries must not be empty");
			}
			const int32_t nameIndex = internInitialStateName(labels[i]);
			m_initialStates.push_back({state, labels[i], weights[i], nameIndex});
		}
	}

	const std::string& initialStateForConstruction() const {
		static const std::string empty;
		if (m_initialStates.empty()) {
			return empty;
		}
		return m_initialStates.front().state;
	}

	py::tuple reset(py::object seedObj = py::none()) {
		if (!seedObj.is_none()) {
			if (PyLong_Check(seedObj.ptr())) {
				const uint64_t seed = seedObj.cast<uint64_t>();
				for (size_t i = 0; i < m_slots.size(); ++i) {
					m_slots[i]->rng.seed(static_cast<uint32_t>(seed + i));
				}
			} else {
				py::sequence seeds = py::reinterpret_borrow<py::sequence>(seedObj);
				if (static_cast<size_t>(seeds.size()) != m_slots.size()) {
					throw std::runtime_error("seed sequence length must match num_envs");
				}
				for (size_t i = 0; i < m_slots.size(); ++i) {
					py::object seed = seeds[static_cast<py::ssize_t>(i)];
					if (!seed.is_none()) {
						m_slots[i]->rng.seed(seed.cast<uint32_t>());
					}
				}
			}
		}
		py::array_t<uint8_t>& obsArray = nextObservationArray();
		uint8_t* obsData = obsArray.mutable_data();
		clearErrors();
		{
			py::gil_scoped_release release;
			batchThreadPool(m_numThreads).parallelFor(m_slots.size(), [&](size_t index) {
				try {
					resetSlot(*m_slots[index], obsData + index * m_stackedObsSize);
				} catch (const std::exception& exc) {
					m_errors[index] = exc.what();
				} catch (...) {
					m_errors[index] = "unknown native vector reset error";
				}
			});
		}
		throwFirstError(m_errors);
		return py::make_tuple(obsArray, resetInfos());
	}

	py::array_t<int32_t> activeStateIndices() {
		return py::array_t<int32_t>(
			{ static_cast<py::ssize_t>(m_activeInitialStateIndices.size()) },
			{ static_cast<py::ssize_t>(sizeof(int32_t)) },
			m_activeInitialStateIndices.data(),
			py::none()
		);
	}

	py::tuple initialStateNames() const {
		py::tuple names(m_initialStateNames.size());
		for (size_t i = 0; i < m_initialStateNames.size(); ++i) {
			names[static_cast<py::ssize_t>(i)] = py::str(m_initialStateNames[i]);
		}
		return names;
	}

	py::tuple initialStatePolicyNames() const {
		py::tuple names(m_initialStates.size());
		for (size_t i = 0; i < m_initialStates.size(); ++i) {
			names[static_cast<py::ssize_t>(i)] = py::str(m_initialStates[i].label);
		}
		return names;
	}

	py::tuple initialStateWeights() const {
		if (m_initialStateMode != InitialStateMode::Sample) {
			throw std::runtime_error("initial_state_weights are only available for weighted state sampling");
		}
		py::tuple weights(m_initialStates.size());
		const double total = m_initialStateCumulative.empty() ? 0.0 : m_initialStateCumulative.back();
		double previous = 0.0;
		for (size_t i = 0; i < m_initialStates.size(); ++i) {
			const double cumulative = m_initialStateCumulative[i];
			const double weight = total > 0.0 ? (cumulative - previous) / total : 0.0;
			weights[static_cast<py::ssize_t>(i)] = weight;
			previous = cumulative;
		}
		return weights;
	}

	void setInitialStates(
		py::object initialStateObj,
		py::object labelsObj = py::none(),
		py::object weightsObj = py::none()
	) {
		parseInitialStates(initialStateObj, labelsObj, weightsObj, m_slots.size());
	}

	void setInitialStateWeights(py::object weightsObj) {
		if (m_initialStateMode != InitialStateMode::Sample) {
			throw std::runtime_error("initial_state_weights can only be updated for weighted state sampling");
		}
		py::sequence weightsSeq = py::reinterpret_borrow<py::sequence>(weightsObj);
		if (static_cast<size_t>(weightsSeq.size()) != m_initialStates.size()) {
			throw std::runtime_error("initial_state_weights length must match initial_state length");
		}

		std::vector<double> weights(m_initialStates.size(), 0.0);
		std::vector<double> cumulativeWeights;
		cumulativeWeights.reserve(m_initialStates.size());
		double cumulative = 0.0;
		for (size_t i = 0; i < m_initialStates.size(); ++i) {
			const double weight = py::float_(weightsSeq[static_cast<py::ssize_t>(i)]).cast<double>();
			if (!std::isfinite(weight) || weight < 0.0) {
				throw std::runtime_error("initial_state_weights must contain non-negative finite numbers");
			}
			weights[i] = weight;
			cumulative += weight;
			cumulativeWeights.push_back(cumulative);
		}
		if (!std::isfinite(cumulative) || cumulative <= 0.0) {
			throw std::runtime_error("initial_state_weights must sum to a positive number");
		}

		for (size_t i = 0; i < weights.size(); ++i) {
			m_initialStates[i].weight = weights[i];
		}
		m_initialStateCumulative = std::move(cumulativeWeights);
	}

	void addStartStateInfo(py::dict& info, const std::string& label) const {
		if (!m_reportInitialState) {
			return;
		}
		info["start_state"] = label;
		info["state"] = label;
	}

	py::list resetInfos() const {
		if (!m_reportInitialState) {
			return m_emptyInfos;
		}
		py::list infos;
		for (const auto& slot : m_slots) {
			py::dict info = py::dict();
			addStartStateInfo(info, slot->currentStartStateLabel);
			infos.append(info);
		}
		return infos;
	}

	py::tuple step(py::array_t<uint8_t> masks) {
		auto mask = masks.unchecked<2>();
		if (static_cast<size_t>(mask.shape(0)) != m_slots.size()) {
			throw std::runtime_error("actions first dimension must match num_envs");
		}
		if (mask.shape(1) != m_numButtons) {
			throw std::runtime_error("actions second dimension must match num_buttons");
		}
		py::array_t<uint8_t>& obsArray = nextObservationArray();
		uint8_t* obsData = obsArray.mutable_data();
		std::vector<StepOutput> outputs(m_slots.size());
		clearErrors();
		clearTerminalObservations();
		{
			py::gil_scoped_release release;
			batchThreadPool(m_numThreads).parallelFor(m_slots.size(), [&](size_t index) {
				try {
					StepOutput output;
					std::vector<uint8_t>& action = m_slots[index]->action;
					for (int key = 0; key < m_numButtons; ++key) {
						action[static_cast<size_t>(key)] = mask(static_cast<py::ssize_t>(index), key) ? 1 : 0;
					}
					stepSlot(*m_slots[index], action, obsData + index * m_stackedObsSize, output, !m_noInfo);
					outputs[index] = std::move(output);
				} catch (const std::exception& exc) {
					m_errors[index] = exc.what();
				} catch (...) {
					m_errors[index] = "unknown native vector step error";
				}
			});
		}
		throwFirstError(m_errors);
		float* rewardData = m_rewardArray.mutable_data();
		bool* doneData = m_doneArray.mutable_data();
		for (size_t i = 0; i < outputs.size(); ++i) {
			rewardData[i] = outputs[i].reward;
			doneData[i] = outputs[i].done;
			if (outputs[i].done) {
				m_terminalObservations[i] = std::move(outputs[i].terminalObservation);
			}
		}
		if (m_noInfo) {
			return py::make_tuple(obsArray, m_rewardArray, m_doneArray, m_emptyInfos);
		}
		py::list infos;
		for (size_t i = 0; i < m_slots.size(); ++i) {
			py::dict info = py::dict();
			if (m_fullInfo || !m_terminalObservations[i].empty()) {
				info = lookupInfo(*m_slots[i]);
			}
			if (m_reportInitialState && (m_fullInfo || !m_terminalObservations[i].empty())) {
				addStartStateInfo(info, outputs[i].startStateLabel);
			}
			if (!m_terminalObservations[i].empty()) {
				py::array_t<uint8_t> terminal(observationShapeContainer());
				std::memcpy(terminal.mutable_data(), m_terminalObservations[i].data(), m_stackedObsSize);
				info["terminal_observation"] = terminal;
				py::dict resetInfo = py::dict();
				addStartStateInfo(resetInfo, outputs[i].resetStartStateLabel);
				info["reset_info"] = resetInfo;
				info["TimeLimit.truncated"] = false;
			}
			if (!outputs[i].firedDoneOnInfoRules.empty()) {
				py::dict doneOnInfo = py::dict();
				for (const DoneOnInfoFiredRule& fired : outputs[i].firedDoneOnInfoRules) {
					py::dict triggerInfo = py::dict();
					triggerInfo["trigger"] = py::str(fired.triggerId);
					triggerInfo["compare"] = py::str(fired.compare);
					triggerInfo["op"] = py::str(doneOnInfoOpName(fired.op));
					py::list keys;
					py::list previousValues;
					py::list currentValues;
					for (size_t keyIndex = 0; keyIndex < fired.keys.size(); ++keyIndex) {
						keys.append(py::str(fired.keys[keyIndex]));
					}
					for (int64_t value : fired.previousValues) {
						previousValues.append(value);
					}
					for (int64_t value : fired.currentValues) {
						currentValues.append(value);
					}
					triggerInfo["keys"] = keys;
					triggerInfo["variables"] = keys;
					triggerInfo["prev"] = previousValues;
					triggerInfo["next"] = currentValues;

					py::str eventName(fired.name);
					if (PyDict_Contains(doneOnInfo.ptr(), eventName.ptr()) == 1) {
						py::dict eventInfo = doneOnInfo[eventName].cast<py::dict>();
						py::list triggers = eventInfo["triggers"].cast<py::list>();
						triggers.append(triggerInfo);
						continue;
					}

					py::dict eventInfo = py::dict();
					eventInfo["trigger"] = triggerInfo["trigger"];
					eventInfo["compare"] = triggerInfo["compare"];
					eventInfo["op"] = triggerInfo["op"];
					eventInfo["keys"] = triggerInfo["keys"];
					eventInfo["variables"] = triggerInfo["variables"];
					eventInfo["prev"] = triggerInfo["prev"];
					eventInfo["next"] = triggerInfo["next"];
					py::list triggers;
					triggers.append(triggerInfo);
					eventInfo["triggers"] = triggers;
					doneOnInfo[eventName] = eventInfo;
				}
				info["done_on_info"] = doneOnInfo;
			}
			infos.append(info);
		}
		return py::make_tuple(obsArray, m_rewardArray, m_doneArray, infos);
	}

	py::tuple observationShape() const {
		if (m_channelsFirst) {
			return py::make_tuple(m_stackedChannels, m_obsHeight, m_obsWidth);
		}
		return py::make_tuple(m_obsHeight, m_obsWidth, m_stackedChannels);
	}

	py::array_t<uint8_t> getScreen(size_t index = 0) {
		if (index >= m_slots.size()) {
			throw std::runtime_error("screen index out of range");
		}
		Slot& slot = *m_slots[index];
		IndexedVideoFrame indexedFrame;
		if (slot.usesIndexedVideo && slot.emulator->m_re.getIndexedVideoFrame(indexedFrame)) {
			py::array_t<uint8_t> screen({
				static_cast<py::ssize_t>(indexedFrame.height),
				static_cast<py::ssize_t>(indexedFrame.width),
				static_cast<py::ssize_t>(3),
			});
			uint8_t* dst = screen.mutable_data();
			for (unsigned y = 0; y < indexedFrame.height; ++y) {
				for (unsigned x = 0; x < indexedFrame.width; ++x) {
					const uint8_t pixel = indexedFrame.data[
						static_cast<size_t>(y) * indexedFrame.pitch + static_cast<size_t>(x)
					];
					uint8_t r = 0;
					uint8_t g = 0;
					uint8_t b = 0;
					rgb565ToRgb888(indexedRgb565(indexedFrame, pixel), r, g, b);
					const size_t offset = (
						static_cast<size_t>(y) * static_cast<size_t>(indexedFrame.width) +
						static_cast<size_t>(x)
					) * 3;
					dst[offset] = r;
					dst[offset + 1] = g;
					dst[offset + 2] = b;
				}
			}
			return screen;
		}
		slot.emulator->readRgbFrame(slot.rgb);
		const long width = slot.emulator->m_re.getImageWidth();
		const long height = slot.emulator->m_re.getImageHeight();
		py::array_t<uint8_t> screen({
			static_cast<py::ssize_t>(height),
			static_cast<py::ssize_t>(width),
			static_cast<py::ssize_t>(3),
		});
		std::memcpy(screen.mutable_data(), slot.rgb.data(), slot.rgb.size());
		return screen;
	}

	size_t numEnvs() const {
		return m_slots.size();
	}

private:
	py::array::ShapeContainer observationShapeContainer() const {
		if (m_channelsFirst) {
			return py::array::ShapeContainer{
				static_cast<py::ssize_t>(m_stackedChannels),
				static_cast<py::ssize_t>(m_obsHeight),
				static_cast<py::ssize_t>(m_obsWidth),
			};
		}
		return py::array::ShapeContainer{
			static_cast<py::ssize_t>(m_obsHeight),
			static_cast<py::ssize_t>(m_obsWidth),
			static_cast<py::ssize_t>(m_stackedChannels),
		};
	}

	py::array::ShapeContainer observationBatchShape() const {
		if (m_channelsFirst) {
			return py::array::ShapeContainer{
				static_cast<py::ssize_t>(m_slots.size()),
				static_cast<py::ssize_t>(m_stackedChannels),
				static_cast<py::ssize_t>(m_obsHeight),
				static_cast<py::ssize_t>(m_obsWidth),
			};
		}
		return py::array::ShapeContainer{
			static_cast<py::ssize_t>(m_slots.size()),
			static_cast<py::ssize_t>(m_obsHeight),
			static_cast<py::ssize_t>(m_obsWidth),
			static_cast<py::ssize_t>(m_stackedChannels),
		};
	}

	py::array_t<uint8_t>& nextObservationArray() {
		if (m_unsafeZeroCopy) {
			return m_obsArrays[0];
		}
		const size_t index = m_nextObsArray;
		m_nextObsArray = 1 - m_nextObsArray;
		return m_obsArrays[index];
	}

	enum class DoneOnInfoOp {
		Change,
		Increase,
		Decrease
	};

	struct DoneOnInfoRule {
		std::string name;
		std::string triggerId = "default";
		std::string compare = "reset";
		DoneOnInfoOp op = DoneOnInfoOp::Change;
		std::vector<std::string> keyNames;
		std::vector<Variable> variables;
	};

	struct DoneOnInfoFiredRule {
		std::string name;
		std::string triggerId = "default";
		std::string compare = "reset";
		DoneOnInfoOp op = DoneOnInfoOp::Change;
		std::vector<std::string> keys;
		std::vector<int64_t> previousValues;
		std::vector<int64_t> currentValues;
	};

	const char* doneOnInfoOpName(DoneOnInfoOp op) const {
		switch (op) {
		case DoneOnInfoOp::Change:
			return "change";
		case DoneOnInfoOp::Increase:
			return "increase";
		case DoneOnInfoOp::Decrease:
			return "decrease";
		}
		return "change";
	}

	DoneOnInfoOp parseDoneOnInfoOp(const std::string& op) const {
		if (op == "change") {
			return DoneOnInfoOp::Change;
		}
		if (op == "increase") {
			return DoneOnInfoOp::Increase;
		}
		if (op == "decrease") {
			return DoneOnInfoOp::Decrease;
		}
		throw std::runtime_error("done_on op must be change, increase, or decrease");
	}

	void parseDoneOnInfoRules(py::object doneOnInfoObj) {
		if (doneOnInfoObj.is_none()) {
			return;
		}
		py::sequence rules = py::reinterpret_borrow<py::sequence>(doneOnInfoObj);
		m_doneOnInfoRules.reserve(static_cast<size_t>(rules.size()));
		for (py::handle rawRule : rules) {
			py::sequence rule = py::reinterpret_borrow<py::sequence>(rawRule);
			const py::ssize_t ruleSize = rule.size();
			if (ruleSize != 3 && ruleSize != 5) {
				throw std::runtime_error("done_on native rules must be (name, keys, op) or (name, trigger, keys, op, compare)");
			}
			DoneOnInfoRule parsed;
			parsed.name = py::str(rule[0]);
			if (parsed.name.empty()) {
				throw std::runtime_error("done_on rule names must not be empty");
			}
			const py::ssize_t triggerIndex = ruleSize == 5 ? 1 : -1;
			const py::ssize_t keysIndex = ruleSize == 5 ? 2 : 1;
			const py::ssize_t opIndex = ruleSize == 5 ? 3 : 2;
			const py::ssize_t compareIndex = ruleSize == 5 ? 4 : -1;
			if (triggerIndex >= 0) {
				parsed.triggerId = py::str(rule[triggerIndex]);
				if (parsed.triggerId.empty()) {
					throw std::runtime_error("done_on trigger ids must not be empty");
				}
			}
			if (compareIndex >= 0) {
				parsed.compare = py::str(rule[compareIndex]);
				if (parsed.compare != "reset") {
					throw std::runtime_error("done_on compare must be reset");
				}
			}
			py::sequence keys = py::reinterpret_borrow<py::sequence>(rule[keysIndex]);
			if (keys.size() == 0) {
				throw std::runtime_error("done_on rules must reference at least one key");
			}
			parsed.keyNames.reserve(static_cast<size_t>(keys.size()));
			for (py::handle rawKey : keys) {
				if (!PyUnicode_Check(rawKey.ptr())) {
					throw std::runtime_error("done_on keys must contain only strings");
				}
				std::string key = py::str(rawKey);
				if (key.empty()) {
					throw std::runtime_error("done_on keys must not be empty");
				}
				parsed.keyNames.push_back(key);
			}
			parsed.op = parseDoneOnInfoOp(py::str(rule[opIndex]));
			m_doneOnInfoRules.push_back(std::move(parsed));
		}
	}

	void resolveDoneOnInfoVariables(const std::unordered_map<std::string, Variable>& variables) {
		for (DoneOnInfoRule& rule : m_doneOnInfoRules) {
			rule.variables.reserve(rule.keyNames.size());
			for (const std::string& key : rule.keyNames) {
				auto variable = variables.find(key);
				if (variable == variables.end()) {
					throw std::runtime_error("unknown done_on key: " + key);
				}
				rule.variables.push_back(variable->second);
			}
		}
	}

	struct Slot {
		Slot(const string& romPath, const string& dataPath, const string& scenarioPath, const std::string& initialState, size_t index)
			: emulator(std::make_unique<PyRetroEmulator>(romPath))
			, index(index)
			, rng(static_cast<uint32_t>(0xC0D3u + index * 9973u)) {
			emulator->configureData(data);
			ScriptContext::reset();
			if (!data.m_data.load(dataPath)) {
				throw std::runtime_error("failed to load data");
			}
			std::vector<Variable> trackedVariables;
			for (const auto& item : data.m_data.listVariables()) {
				trackedVariables.push_back(item.second);
			}
			data.m_data.setTrackedVariables(trackedVariables);
			if (!data.m_scen.load(scenarioPath)) {
				throw std::runtime_error("failed to load scenario");
			}
			if (!initialState.empty() && !emulator->m_re.unserialize(initialState.data(), initialState.size())) {
				throw std::runtime_error("failed to load initial state");
			}
			emulator->m_re.run();
			data.reset();
			data.updateRam();
		}

		std::unique_ptr<PyRetroEmulator> emulator;
		size_t index = 0;
		PyGameData data;
		std::vector<uint8_t> frameStack;
		std::vector<uint8_t> singleObs;
		std::vector<uint8_t> rgb;
		std::vector<uint8_t> prevRaw;
		std::vector<uint8_t> currRaw;
		std::vector<uint8_t> prevIndexed;
		IndexedPaletteCache indexedPaletteCache;
		std::vector<uint8_t> action;
		std::vector<uint8_t> lastMask;
		bool hasLastMask = false;
		bool usesIndexedVideo = false;
		std::vector<std::vector<int64_t>> doneOnInfoBaselines;
		std::string currentStartStateLabel;
		std::mt19937 rng;
	};

	struct StepOutput {
		float reward = 0.0f;
		bool done = false;
		std::vector<DoneOnInfoFiredRule> firedDoneOnInfoRules;
		std::string startStateLabel;
		std::string resetStartStateLabel;
		std::vector<uint8_t> terminalObservation;
	};

	py::dict lookupInfo(const Slot& slot) const {
		if (m_infoKeys.empty()) {
			return slot.data.lookupAll();
		}
		py::dict data;
		for (const auto& item : m_infoVariables) {
			data[py::str(item.first)] = slot.data.m_data.lookupValue(item.second);
		}
		return data;
	}

	void clearKeys(Slot& slot) {
		for (int key = 0; key < m_numButtons; ++key) {
			slot.emulator->m_re.setKey(0, key, false);
		}
	}

	void setKeys(Slot& slot, const std::vector<uint8_t>& mask) {
		uint16_t actionBits = 0;
		for (int key = 0; key < m_numButtons; ++key) {
			if (mask[static_cast<size_t>(key)]) {
				actionBits |= static_cast<uint16_t>(1u << key);
			}
		}
		if (m_filterActions) {
			actionBits = slot.data.m_scen.filterAction(actionBits);
		}
		for (int key = 0; key < m_numButtons; ++key) {
			slot.emulator->m_re.setKey(0, key, (actionBits >> key) & 1u);
		}
	}

	void readProcessedFrame(Slot& slot) {
		slot.singleObs.resize(m_singleObsSize);
		if (m_grayscale) {
			IndexedVideoFrame indexedFrame;
			if (slot.usesIndexedVideo && slot.emulator->m_re.getIndexedVideoFrame(indexedFrame) && m_useGrayscaleAreaPlan) {
				processIndexedGrayscaleAreaPlanToBuffer(
					indexedFrame,
					nullptr,
					&slot.indexedPaletteCache,
					m_grayscaleAreaPlan,
					slot.singleObs.data()
				);
				return;
			}
			const uint8_t* raw = static_cast<const uint8_t*>(slot.emulator->m_re.getImageData());
			if (!raw) {
				slot.emulator->readRawFrame(slot.currRaw);
				raw = slot.currRaw.data();
			}
			if (m_useGrayscaleAreaPlan) {
				processNativeGrayscaleAreaPlanToBuffer(
					raw,
					nullptr,
					static_cast<size_t>(slot.emulator->m_re.getImagePitch()),
					slot.emulator->m_re.getImageDepth(),
					m_grayscaleAreaPlan,
					slot.singleObs.data()
				);
				return;
			}
			processNativeGrayscaleFrameToBuffer(
				raw,
				nullptr,
				slot.emulator->m_re.getImageWidth(),
				slot.emulator->m_re.getImageHeight(),
				static_cast<size_t>(slot.emulator->m_re.getImagePitch()),
				slot.emulator->m_re.getImageDepth(),
				m_crop,
				m_cropMask,
				m_resize,
				m_algorithm,
				m_cropFill,
				slot.singleObs.data()
			);
			return;
		}
		slot.emulator->readRgbFrame(slot.rgb);
		processRgbFrameToBuffer(
			slot.rgb,
			slot.emulator->m_re.getImageWidth(),
			slot.emulator->m_re.getImageHeight(),
			m_crop,
			m_cropMask,
			m_resize,
			m_grayscale,
			m_algorithm,
			m_cropFill,
			slot.singleObs.data()
		);
	}

	void resetFrameStack(Slot& slot, const std::vector<uint8_t>& singleObs) {
		if (m_channelsFirst) {
			const size_t frameSize = m_singleObsSize;
			for (int frame = 0; frame < m_frameStack; ++frame) {
				writeSingleObservationChannelsFirst(
					singleObs,
					slot.frameStack.data() + static_cast<size_t>(frame) * frameSize
				);
			}
			return;
		}
		const size_t pixelCount = static_cast<size_t>(m_obsHeight) * static_cast<size_t>(m_obsWidth);
		const size_t channels = static_cast<size_t>(m_obsChannels);
		const size_t stackedChannels = static_cast<size_t>(m_stackedChannels);
		for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
			const uint8_t* src = singleObs.data() + pixel * channels;
			uint8_t* dst = slot.frameStack.data() + pixel * stackedChannels;
			for (int frame = 0; frame < m_frameStack; ++frame) {
				std::memcpy(dst + static_cast<size_t>(frame) * channels, src, channels);
			}
		}
	}

	void pushFrame(Slot& slot, const std::vector<uint8_t>& singleObs) {
		if (m_channelsFirst) {
			const size_t frameSize = m_singleObsSize;
			if (m_frameStack > 1) {
				std::memmove(
					slot.frameStack.data(),
					slot.frameStack.data() + frameSize,
					static_cast<size_t>(m_frameStack - 1) * frameSize
				);
			}
			writeSingleObservationChannelsFirst(
				singleObs,
				slot.frameStack.data() + static_cast<size_t>(m_frameStack - 1) * frameSize
			);
			return;
		}
		if (m_frameStack == 1) {
			std::memcpy(slot.frameStack.data(), singleObs.data(), m_singleObsSize);
			return;
		}
		const size_t pixelCount = static_cast<size_t>(m_obsHeight) * static_cast<size_t>(m_obsWidth);
		const size_t channels = static_cast<size_t>(m_obsChannels);
		const size_t stackedChannels = static_cast<size_t>(m_stackedChannels);
		const size_t keptChannels = static_cast<size_t>(m_frameStack - 1) * channels;
		for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
			uint8_t* dst = slot.frameStack.data() + pixel * stackedChannels;
			std::memmove(dst, dst + channels, keptChannels);
			std::memcpy(dst + keptChannels, singleObs.data() + pixel * channels, channels);
		}
	}

	void writeStackedObservation(const Slot& slot, uint8_t* dst) const {
		std::memcpy(dst, slot.frameStack.data(), m_stackedObsSize);
	}

	void writeSingleObservationChannelsFirst(const std::vector<uint8_t>& singleObs, uint8_t* dst) const {
		const size_t pixelCount = static_cast<size_t>(m_obsHeight) * static_cast<size_t>(m_obsWidth);
		const size_t channels = static_cast<size_t>(m_obsChannels);
		if (channels == 1) {
			std::memcpy(dst, singleObs.data(), pixelCount);
			return;
		}
		for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
			const uint8_t* src = singleObs.data() + pixel * channels;
			for (size_t channel = 0; channel < channels; ++channel) {
				dst[channel * pixelCount + pixel] = src[channel];
			}
		}
	}

	float clipReward(float reward) const {
		if (!m_rewardClip) {
			return reward;
		}
		return std::max(m_rewardClipLow, std::min(m_rewardClipHigh, reward));
	}

	int32_t initialStateIndexForReset(Slot& slot) {
		switch (m_initialStateMode) {
		case InitialStateMode::None:
			return -1;
		case InitialStateMode::Single:
			return m_initialStates.empty() ? -1 : 0;
		case InitialStateMode::Slot:
			return static_cast<int32_t>(slot.index);
		case InitialStateMode::Sample: {
			std::uniform_real_distribution<double> dist(0.0, m_initialStateCumulative.back());
			const double sample = dist(slot.rng);
			auto it = std::upper_bound(m_initialStateCumulative.begin(), m_initialStateCumulative.end(), sample);
			size_t index = static_cast<size_t>(std::distance(m_initialStateCumulative.begin(), it));
			if (index >= m_initialStates.size()) {
				index = m_initialStates.size() - 1;
			}
			return static_cast<int32_t>(index);
		}
		}
		return -1;
	}

	void resetSlot(Slot& slot, uint8_t* dst) {
		const int32_t initialStateIndex = initialStateIndexForReset(slot);
		const InitialStateChoice* initialState =
			initialStateIndex >= 0 ? &m_initialStates[static_cast<size_t>(initialStateIndex)] : nullptr;
		if (initialState && !initialState->state.empty()) {
			if (!slot.emulator->m_re.unserialize(initialState->state.data(), initialState->state.size())) {
				throw std::runtime_error("failed to load initial state");
			}
			slot.currentStartStateLabel = initialState->label;
			m_activeInitialStateIndices[slot.index] = initialState->nameIndex;
		} else {
			slot.emulator->m_re.reset();
			slot.currentStartStateLabel.clear();
			m_activeInitialStateIndices[slot.index] = -1;
		}
		slot.hasLastMask = false;
		std::fill(slot.lastMask.begin(), slot.lastMask.end(), 0);
		clearKeys(slot);
		slot.emulator->m_re.run();
		slot.data.reset();
		slot.data.updateRam();
		if (m_noopResetMax > 0) {
			std::uniform_int_distribution<int> noopDist(0, m_noopResetMax);
			const int noopCount = noopDist(slot.rng);
			for (int i = 0; i < noopCount; ++i) {
				slot.emulator->m_re.run();
				slot.data.m_data.updateRam();
				slot.data.m_scen.update(1);
				if (slot.data.m_scen.isDone()) {
					break;
				}
			}
		}
		readProcessedFrame(slot);
		resetFrameStack(slot, slot.singleObs);
		writeStackedObservation(slot, dst);
		initializeDoneOnInfoBaselines(slot);
	}

	void initializeDoneOnInfoBaselines(Slot& slot) {
		if (m_doneOnInfoRules.empty()) {
			return;
		}
		slot.doneOnInfoBaselines.resize(m_doneOnInfoRules.size());
		for (size_t ruleIndex = 0; ruleIndex < m_doneOnInfoRules.size(); ++ruleIndex) {
			const DoneOnInfoRule& rule = m_doneOnInfoRules[ruleIndex];
			std::vector<int64_t>& baseline = slot.doneOnInfoBaselines[ruleIndex];
			baseline.resize(rule.variables.size());
			for (size_t keyIndex = 0; keyIndex < rule.variables.size(); ++keyIndex) {
				baseline[keyIndex] = slot.data.m_data.lookupValue(rule.variables[keyIndex]);
			}
		}
	}

	bool doneOnInfoValueFired(DoneOnInfoOp op, int64_t baseline, int64_t current) const {
		switch (op) {
		case DoneOnInfoOp::Change:
			return current != baseline;
		case DoneOnInfoOp::Increase:
			return current > baseline;
		case DoneOnInfoOp::Decrease:
			return current < baseline;
		}
		return false;
	}

	bool checkDoneOnInfo(Slot& slot, StepOutput& output) {
		if (m_doneOnInfoRules.empty()) {
			return false;
		}
		bool firedAny = false;
		for (size_t ruleIndex = 0; ruleIndex < m_doneOnInfoRules.size(); ++ruleIndex) {
			const DoneOnInfoRule& rule = m_doneOnInfoRules[ruleIndex];
			if (slot.doneOnInfoBaselines[ruleIndex].size() != rule.variables.size()) {
				initializeDoneOnInfoBaselines(slot);
			}
			const std::vector<int64_t>& baseline = slot.doneOnInfoBaselines[ruleIndex];
			bool fired = false;
			std::vector<int64_t> currentValues(rule.variables.size());
			for (size_t keyIndex = 0; keyIndex < rule.variables.size(); ++keyIndex) {
				const int64_t current = slot.data.m_data.lookupValue(rule.variables[keyIndex]);
				currentValues[keyIndex] = current;
				if (doneOnInfoValueFired(rule.op, baseline[keyIndex], current)) {
					fired = true;
				}
			}
			if (!fired) {
				continue;
			}
			firedAny = true;
			output.firedDoneOnInfoRules.push_back({
				rule.name,
				rule.triggerId,
				rule.compare,
				rule.op,
				rule.keyNames,
				baseline,
				std::move(currentValues),
			});
		}
		return firedAny;
	}

	void stepSlot(Slot& slot, const std::vector<uint8_t>& requestedAction, uint8_t* dst, StepOutput& output, bool captureTerminalObservation) {
		const std::vector<uint8_t>* action = &requestedAction;
		if (m_stickyActionProb > 0.0) {
			bool repeatPreviousAction = false;
			if (slot.hasLastMask) {
				std::uniform_real_distribution<double> stickyDist(0.0, 1.0);
				if (stickyDist(slot.rng) < m_stickyActionProb) {
					repeatPreviousAction = true;
				}
			}
			if (repeatPreviousAction) {
				action = &slot.lastMask;
			} else {
				slot.lastMask.assign(requestedAction.begin(), requestedAction.end());
			}
			slot.hasLastMask = true;
		}
		setKeys(slot, *action);

		bool done = false;
		float totalReward = 0.0f;
		bool sawFrame = false;
		std::vector<uint8_t> prevRgb;
		std::vector<uint8_t> currRgb;
		std::vector<uint8_t> preSkipState;
		slot.prevRaw.clear();
		slot.prevIndexed.clear();
		for (int i = 0; i < m_frameSkip; ++i) {
			const bool mayNeedPixels =
				m_maxpoolLastTwo ? (i >= m_frameSkip - 2) : (i == m_frameSkip - 1);
			bool skippedRender = false;
			if (m_renderSkipEnabled && !mayNeedPixels) {
				if (captureTerminalObservation) {
					const size_t stateSize = slot.emulator->m_re.serializeSize();
					preSkipState.resize(stateSize);
					if (!slot.emulator->m_re.serialize(preSkipState.data(), stateSize)) {
						throw std::runtime_error("failed to serialize pre-render-skip state");
					}
				}
				skippedRender = slot.emulator->m_re.runSkipRender();
			} else {
				slot.emulator->m_re.run();
			}
			slot.data.m_data.updateRam();
			slot.data.m_scen.update(1);
			totalReward += slot.data.m_scen.currentReward();
			const bool scenarioDone = slot.data.m_scen.isDone();
			const bool doneOnInfo = checkDoneOnInfo(slot, output);
			done = scenarioDone || doneOnInfo;
			if (done && skippedRender && captureTerminalObservation) {
				if (!slot.emulator->m_re.unserialize(preSkipState.data(), preSkipState.size())) {
					throw std::runtime_error("failed to restore pre-render-skip terminal state");
				}
				slot.emulator->m_re.run();
				slot.data.m_data.updateRam();
			}
			if (m_maxpoolLastTwo && (i >= m_frameSkip - 2 || done)) {
				if (m_grayscale) {
					if (!done && i < m_frameSkip - 1) {
						IndexedVideoFrame indexedFrame;
						if (slot.usesIndexedVideo && slot.emulator->m_re.getIndexedVideoFrame(indexedFrame)) {
							slot.prevIndexed.resize(static_cast<size_t>(indexedFrame.pitch) * static_cast<size_t>(indexedFrame.height));
							for (unsigned row = 0; row < indexedFrame.height; ++row) {
								std::memcpy(
									slot.prevIndexed.data() + static_cast<size_t>(row) * indexedFrame.pitch,
									indexedFrame.data + static_cast<size_t>(row) * indexedFrame.pitch,
									indexedFrame.pitch
								);
							}
						} else {
							slot.emulator->readRawFrame(slot.prevRaw);
						}
					}
				} else {
					prevRgb.swap(currRgb);
					slot.emulator->readRgbFrame(currRgb);
				}
				sawFrame = true;
			}
			if (done) {
				break;
			}
		}

		slot.singleObs.resize(m_singleObsSize);
		if (!m_maxpoolLastTwo || !sawFrame) {
			readProcessedFrame(slot);
		} else {
			if (m_grayscale) {
				IndexedVideoFrame indexedFrame;
				if (slot.usesIndexedVideo && slot.emulator->m_re.getIndexedVideoFrame(indexedFrame) && m_useGrayscaleAreaPlan) {
					processIndexedGrayscaleAreaPlanToBuffer(
						indexedFrame,
						slot.prevIndexed.empty() ? nullptr : slot.prevIndexed.data(),
						&slot.indexedPaletteCache,
						m_grayscaleAreaPlan,
						slot.singleObs.data()
					);
					pushFrame(slot, slot.singleObs);
					output.reward = clipReward(totalReward);
					output.done = done;
					if (m_reportInitialState && (m_fullInfo || done)) {
						output.startStateLabel = slot.currentStartStateLabel;
					}
					if (done && captureTerminalObservation) {
						output.terminalObservation.resize(m_stackedObsSize);
						writeStackedObservation(slot, output.terminalObservation.data());
						resetSlot(slot, dst);
						output.resetStartStateLabel = slot.currentStartStateLabel;
					} else if (done) {
						resetSlot(slot, dst);
						output.resetStartStateLabel = slot.currentStartStateLabel;
					} else {
						writeStackedObservation(slot, dst);
					}
					return;
				}
				const uint8_t* raw = static_cast<const uint8_t*>(slot.emulator->m_re.getImageData());
				if (!raw) {
					slot.emulator->readRawFrame(slot.currRaw);
					raw = slot.currRaw.data();
				}
				if (m_useGrayscaleAreaPlan) {
					processNativeGrayscaleAreaPlanToBuffer(
						raw,
						slot.prevRaw.empty() ? nullptr : slot.prevRaw.data(),
						static_cast<size_t>(slot.emulator->m_re.getImagePitch()),
						slot.emulator->m_re.getImageDepth(),
						m_grayscaleAreaPlan,
						slot.singleObs.data()
					);
				} else {
					processNativeGrayscaleFrameToBuffer(
						raw,
						slot.prevRaw.empty() ? nullptr : slot.prevRaw.data(),
						slot.emulator->m_re.getImageWidth(),
						slot.emulator->m_re.getImageHeight(),
						static_cast<size_t>(slot.emulator->m_re.getImagePitch()),
						slot.emulator->m_re.getImageDepth(),
						m_crop,
						m_cropMask,
						m_resize,
						m_algorithm,
						m_cropFill,
						slot.singleObs.data()
					);
				}
			} else {
				if (!prevRgb.empty()) {
					for (size_t i = 0; i < currRgb.size(); ++i) {
						currRgb[i] = std::max(currRgb[i], prevRgb[i]);
					}
				}
				processRgbFrameToBuffer(
					currRgb,
					slot.emulator->m_re.getImageWidth(),
					slot.emulator->m_re.getImageHeight(),
					m_crop,
					m_cropMask,
					m_resize,
					m_grayscale,
					m_algorithm,
					m_cropFill,
					slot.singleObs.data()
				);
			}
		}
		pushFrame(slot, slot.singleObs);
		output.reward = clipReward(totalReward);
		output.done = done;
		if (m_reportInitialState && (m_fullInfo || done)) {
			output.startStateLabel = slot.currentStartStateLabel;
		}
		if (done && captureTerminalObservation) {
			output.terminalObservation.resize(m_stackedObsSize);
			writeStackedObservation(slot, output.terminalObservation.data());
			resetSlot(slot, dst);
			output.resetStartStateLabel = slot.currentStartStateLabel;
		} else if (done) {
			resetSlot(slot, dst);
			output.resetStartStateLabel = slot.currentStartStateLabel;
		} else {
			writeStackedObservation(slot, dst);
		}
	}

	void clearErrors() {
		for (std::string& error : m_errors) {
			error.clear();
		}
	}

	void clearTerminalObservations() {
		for (std::vector<uint8_t>& terminalObservation : m_terminalObservations) {
			terminalObservation.clear();
		}
	}

	void throwFirstError(const std::vector<std::string>& errors) {
		for (const auto& error : errors) {
			if (!error.empty()) {
				throw std::runtime_error(error);
			}
		}
	}

	std::vector<std::unique_ptr<Slot>> m_slots;
	std::vector<InitialStateChoice> m_initialStates;
	std::vector<std::string> m_initialStateNames;
	std::vector<double> m_initialStateCumulative;
	std::vector<int32_t> m_activeInitialStateIndices;
	InitialStateMode m_initialStateMode = InitialStateMode::None;
	bool m_reportInitialState = false;
	int m_numButtons = 0;
	int m_frameSkip = 1;
	int m_frameStack = 1;
	NativeCrop m_crop;
	NativeCrop m_cropMask;
	NativeResize m_resize;
	uint8_t m_cropFill = 0;
	bool m_grayscale = false;
	string m_algorithm;
	bool m_maxpoolLastTwo = false;
	int m_noopResetMax = 0;
	double m_stickyActionProb = 0.0;
	bool m_filterActions = false;
	bool m_rewardClip = false;
	float m_rewardClipLow = -1.0f;
	float m_rewardClipHigh = 1.0f;
	bool m_fullInfo = false;
	bool m_noInfo = false;
	bool m_unsafeZeroCopy = false;
	bool m_channelsFirst = false;
	bool m_renderSkipEnabled = false;
	std::vector<DoneOnInfoRule> m_doneOnInfoRules;
	std::vector<std::string> m_infoKeys;
	std::vector<std::pair<std::string, Variable>> m_infoVariables;
	int m_numThreads = 1;
	long m_obsHeight = 0;
	long m_obsWidth = 0;
	int m_obsChannels = 0;
	int m_stackedChannels = 0;
	size_t m_singleObsSize = 0;
	size_t m_stackedObsSize = 0;
	bool m_useGrayscaleAreaPlan = false;
	AreaResizePlan m_grayscaleAreaPlan;
	std::array<py::array_t<uint8_t>, 2> m_obsArrays;
	size_t m_nextObsArray = 0;
	py::array_t<float> m_rewardArray;
	py::array_t<bool> m_doneArray;
	py::list m_emptyInfos;
	std::vector<std::string> m_errors;
	std::vector<std::vector<uint8_t>> m_terminalObservations;
};

	py::tuple PyRetroEmulator::stepRepeatAndProcess(
		PyGameData& data,
		py::array_t<uint8_t> mask,
		int repeats,
		py::object cropObj,
		py::object resizeObj,
		bool grayscale,
		const string& algorithm,
		bool maxpoolLastTwo,
		py::object maskCropObj,
		int cropFill
	) {
	if (repeats <= 0) {
		throw std::runtime_error("repeats must be positive");
	}
	if (mask.size() > N_BUTTONS) {
		throw std::runtime_error("mask.size() > N_BUTTONS");
	}
	for (int key = 0; key < mask.size(); ++key) {
		m_re.setKey(0, key, mask.data()[key]);
	}

	float totalReward = 0.0f;
	bool done = false;
	bool sawFrame = false;
	std::vector<uint8_t> prevRgb;
	std::vector<uint8_t> currRgb;

	{
		py::gil_scoped_release release;
		for (int i = 0; i < repeats; ++i) {
			m_re.run();
			data.m_data.updateRam();
			data.m_scen.update();
			totalReward += data.m_scen.currentReward();
			done = data.m_scen.isDone();
			if (maxpoolLastTwo && (i >= repeats - 2 || done)) {
				prevRgb.swap(currRgb);
				readRgbFrame(currRgb);
				sawFrame = true;
			}
			if (done) {
				break;
			}
		}

		if (!maxpoolLastTwo || !sawFrame) {
			readRgbFrame(currRgb);
		} else if (!prevRgb.empty()) {
			for (size_t i = 0; i < currRgb.size(); ++i) {
				currRgb[i] = std::max(currRgb[i], prevRgb[i]);
			}
		}
	}

	py::array_t<uint8_t> obs = processRgbFrame(
		currRgb,
		cropObj,
		resizeObj,
		grayscale,
		algorithm,
		maskCropObj,
		cropFill
	);
	return py::make_tuple(obs, totalReward, done, data.lookupAll());
}

struct PyMovie {
	std::unique_ptr<Retro::Movie> m_movie;
	bool recording = false;
	PyMovie(py::str name, bool record, unsigned players) {
		recording = record;
		if (record) {
			m_movie = std::make_unique<MovieBK2>(name, true, players);
		} else {
			m_movie = Movie::load(name);
		}
		if (!m_movie) {
			throw std::runtime_error("Could not load movie");
		}
	}

	void configure(py::str name, const PyRetroEmulator& emu) {
		if (recording) {
			static_cast<MovieBK2*>(m_movie.get())->setGameName(name);
			static_cast<MovieBK2*>(m_movie.get())->loadKeymap(emu.m_re.core());
		}
	}

	py::str getGameName() const {
		return m_movie->getGameName();
	}

	bool step() {
		return m_movie->step();
	}

	void close() {
		m_movie->close();
	}

	unsigned players() {
		return m_movie->players();
	}

	bool getKey(int key, unsigned player = 0) {
		return m_movie->getKey(key, player);
	}

	void setKey(int key, bool set, unsigned player = 0) {
		return m_movie->setKey(key, set, player);
	}

	py::bytes getState() {
		std::vector<uint8_t> data;
		m_movie->getState(&data);
		return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
	}

	void setState(py::bytes data) {
		m_movie->setState(reinterpret_cast<uint8_t*>(PyBytes_AsString(data.ptr())), PyBytes_Size(data.ptr()));
	}
};

py::str corePath(py::handle hint = py::none()) {
	return Retro::corePath(py::str(hint));
}

py::str dataPath(py::handle hint = py::none()) {
	return Retro::GameData::dataPath(py::str(hint));
}

PYBIND11_MODULE(_retro, m) {
	m.doc() = "libretro bindings";

	py::class_<PyRetroEmulator>(m, "RetroEmulator")
		.def(py::init<const string&>())
		.def("step", &PyRetroEmulator::step, py::call_guard<py::gil_scoped_release>())
		.def("set_button_mask", &PyRetroEmulator::setButtonMask, py::arg("mask"), py::arg("player") = 0)
		.def("get_state", &PyRetroEmulator::getState)
		.def("set_state", &PyRetroEmulator::setState)
		.def("get_screen", &PyRetroEmulator::getScreen)
		.def(
			"get_processed_screen",
			&PyRetroEmulator::getProcessedScreen,
			py::arg("crop"),
			py::arg("resize"),
			py::arg("grayscale"),
			py::arg("algorithm"),
			py::arg("mask_crop") = py::none(),
			py::arg("crop_fill") = 0
		)
		.def(
			"step_repeat_and_process",
			&PyRetroEmulator::stepRepeatAndProcess,
			py::arg("data"),
			py::arg("mask"),
			py::arg("repeats"),
			py::arg("crop"),
			py::arg("resize"),
			py::arg("grayscale"),
			py::arg("algorithm"),
			py::arg("maxpool_last_two"),
			py::arg("mask_crop") = py::none(),
			py::arg("crop_fill") = 0
		)
		.def("get_rotation", &PyRetroEmulator::getRotation)
		.def("get_screen_rate", &PyRetroEmulator::getScreenRate)
		.def("get_audio", &PyRetroEmulator::getAudio)
		.def("get_audio_rate", &PyRetroEmulator::getAudioRate)
		.def("get_resolution", &PyRetroEmulator::getResolution)
		.def("configure_data", &PyRetroEmulator::configureData)
		.def("add_cheat", &PyRetroEmulator::addCheat)
		.def("clear_cheats", &PyRetroEmulator::clearCheats)
		.def_static("load_core_info", &PyRetroEmulator::loadCoreInfo);

	py::class_<PyMemoryView>(m, "Memory")
		.def(py::init<Retro::AddressSpace&>())
		.def("extract", &PyMemoryView::extract, py::arg("address"), py::arg("type"))
		.def("assign", &PyMemoryView::assign, py::arg("address"), py::arg("type"), py::arg("value"))
		.def_property_readonly("blocks", &PyMemoryView::blocks)
		.def("__setitem__", &PyMemoryView::setitem, py::arg("item"), py::arg("value"))
		.def("__getitem__", &PyMemoryView::getitem, py::arg("item"));

	py::class_<PySearch>(m, "Search")
		.def(py::init<py::handle>(), py::arg("types") = py::none())
		.def("num_results", &PySearch::numResults)
		.def("has_unique_result", &PySearch::hasUniqueResult)
		.def("unique_result", &PySearch::uniqueResult)
		.def("typed_results", &PySearch::typedResults);

	py::class_<PyGameData>(m, "GameDataGlue")
		.def(py::init<>())
		.def("load", &PyGameData::load, py::arg("data") = py::none(), py::arg("scen") = py::none())
		.def("save", &PyGameData::save, py::arg("data") = py::none(), py::arg("scen") = py::none())
		.def("reset", &PyGameData::reset)
		.def("filter_action", &PyGameData::filterAction)
		.def("valid_actions", &PyGameData::validActions)
		.def("update_ram", &PyGameData::updateRam)
		.def("lookup_value", &PyGameData::lookupValue)
		.def("set_value", &PyGameData::setValue)
		.def("lookup_all", &PyGameData::lookupAll)
		.def("get_variable", &PyGameData::getVariable)
		.def("set_variable", &PyGameData::setVariable)
		.def("remove_variable", &PyGameData::removeVariable)
		.def("list_variables", &PyGameData::listVariables)
		.def("search", &PyGameData::search)
		.def("delta_search", &PyGameData::deltaSearch)
		.def("get_search", &PyGameData::getSearch)
		.def("remove_search", &PyGameData::removeSearch)
		.def("list_searches", &PyGameData::listSearches)
		.def("current_reward", &PyGameData::currentReward, py::arg("player") = 0)
		.def("total_reward", &PyGameData::totalReward, py::arg("player") = 0)
		.def("is_done", &PyGameData::isDone)
		.def("crop_info", &PyGameData::cropInfo, py::arg("player") = 0)
		.def_property_readonly("memory", &PyGameData::memory);

	py::class_<PyMovie>(m, "Movie")
		.def(py::init<py::str, bool, unsigned>(), py::arg("path"), py::arg("record") = false, py::arg("players") = 1)
		.def("configure", &PyMovie::configure)
		.def("get_game", &PyMovie::getGameName)
		.def("step", &PyMovie::step)
		.def("close", &PyMovie::close)
		.def_property_readonly("players", &PyMovie::players)
		.def("get_key", &PyMovie::getKey)
		.def("set_key", &PyMovie::setKey)
		.def("get_state", &PyMovie::getState)
		.def("set_state", &PyMovie::setState);

	py::class_<PyRetroVecEnv>(m, "_RetroVecEnv")
		.def(
			py::init<
				size_t,
				const string&,
				const string&,
				const string&,
				py::object,
				int,
				int,
				int,
				py::object,
				py::object,
				bool,
				const string&,
				bool,
				int,
				double,
				bool,
				bool,
				float,
				float,
				int,
				const string&,
				bool,
				const string&,
				py::object,
				py::object,
				py::object,
				py::object,
				py::object,
				int>(),
			py::arg("num_envs"),
			py::arg("rom_path"),
			py::arg("data_path"),
			py::arg("scenario_path"),
			py::arg("initial_state"),
			py::arg("num_buttons"),
			py::arg("frame_skip"),
			py::arg("frame_stack"),
			py::arg("crop"),
			py::arg("resize"),
			py::arg("grayscale"),
			py::arg("algorithm"),
			py::arg("maxpool_last_two"),
			py::arg("noop_reset_max"),
			py::arg("sticky_action_prob"),
			py::arg("filter_actions"),
			py::arg("reward_clip"),
			py::arg("reward_clip_low"),
			py::arg("reward_clip_high"),
			py::arg("num_threads") = 0,
			py::arg("info_filter_mode") = "all",
			py::arg("unsafe_view") = false,
			py::arg("obs_layout") = "hwc",
			py::arg("info_filter_keys") = py::none(),
			py::arg("done_on_rules") = py::none(),
			py::arg("initial_state_labels") = py::none(),
			py::arg("initial_state_weights") = py::none(),
			py::arg("mask_crop") = py::none(),
			py::arg("crop_fill") = 0
		)
		.def("reset", &PyRetroVecEnv::reset, py::arg("seed") = py::none())
		.def("step", &PyRetroVecEnv::step)
		.def("active_state_indices", &PyRetroVecEnv::activeStateIndices)
		.def("initial_state_policy_names", &PyRetroVecEnv::initialStatePolicyNames)
		.def("initial_state_weights", &PyRetroVecEnv::initialStateWeights)
		.def(
			"set_initial_states",
			&PyRetroVecEnv::setInitialStates,
			py::arg("initial_state"),
			py::arg("initial_state_labels") = py::none(),
			py::arg("initial_state_weights") = py::none()
		)
		.def("set_initial_state_weights", &PyRetroVecEnv::setInitialStateWeights)
		.def("observation_shape", &PyRetroVecEnv::observationShape)
		.def("get_screen", &PyRetroVecEnv::getScreen, py::arg("index") = 0)
		.def_property_readonly("initial_state_names", &PyRetroVecEnv::initialStateNames)
		.def_property_readonly("num_envs", &PyRetroVecEnv::numEnvs);

	m.def("core_path", &::corePath, py::arg("hint") = py::none());
	m.def("data_path", &::dataPath, py::arg("hint") = py::none());
	}
