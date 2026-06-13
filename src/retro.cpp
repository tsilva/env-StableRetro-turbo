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
#include <atomic>
#include <condition_variable>
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

	py::array_t<uint8_t> processRgbFrame(const std::vector<uint8_t>& rgb, py::object cropObj, py::object resizeObj, bool grayscale, const string& algorithm) {
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

		auto srcPixel = [&](long y, long x, int c) -> uint8_t {
			const size_t offset = (static_cast<size_t>(top + y) * static_cast<size_t>(w) + static_cast<size_t>(left + x)) * 3;
			return rgb[offset + c];
		};
		auto srcGray = [&](long y, long x) -> uint8_t {
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

	py::array_t<uint8_t> getProcessedScreen(py::object cropObj, py::object resizeObj, bool grayscale, const string& algorithm) {
		std::vector<uint8_t> rgb;
		readRgbFrame(rgb);
		return processRgbFrame(rgb, cropObj, resizeObj, grayscale, algorithm);
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
	py::tuple stepRepeatAndProcess(PyGameData& data, py::array_t<uint8_t> mask, int repeats, py::object cropObj, py::object resizeObj, bool grayscale, const string& algorithm, bool maxpoolLastTwo);
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

	py::object lookupValue(py::str name) const {
		try {
			Variant data = m_data.lookupValue(name);
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

void processRgbFrameToBuffer(
	const std::vector<uint8_t>& rgb,
	long rawW,
	long rawH,
	const NativeCrop& crop,
	const NativeResize& resize,
	bool grayscale,
	const string& algorithm,
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
		const size_t offset = (static_cast<size_t>(crop.top + y) * static_cast<size_t>(rawW) + static_cast<size_t>(crop.left + x)) * 3;
		return rgb[offset + c];
	};
	auto srcGray = [&](long y, long x) -> uint8_t {
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

class PyNativeVectorEnv {
public:
	PyNativeVectorEnv(
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
		int numThreads
	)
		: m_numButtons(numButtons)
		, m_frameSkip(frameSkip)
		, m_frameStack(frameStack)
		, m_crop(parseNativeCrop(cropObj))
		, m_resize(parseNativeResize(resizeObj))
		, m_grayscale(grayscale)
		, m_algorithm(algorithm)
		, m_maxpoolLastTwo(maxpoolLastTwo)
		, m_noopResetMax(noopResetMax)
		, m_stickyActionProb(stickyActionProb)
		, m_filterActions(filterActions)
		, m_rewardClip(rewardClip)
		, m_rewardClipLow(rewardClipLow)
		, m_rewardClipHigh(rewardClipHigh)
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
		if (algorithm != "nearest" && algorithm != "bilinear" && algorithm != "area") {
			throw std::runtime_error("algorithm must be nearest, bilinear, or area");
		}
		if (!initialStateObj.is_none()) {
			m_initialState = py::bytes(initialStateObj);
		}
		m_numThreads = std::max(1, std::min<int>(m_numThreads <= 0 ? static_cast<int>(numEnvs) : m_numThreads, static_cast<int>(numEnvs)));
		m_slots.reserve(numEnvs);
		for (size_t i = 0; i < numEnvs; ++i) {
			auto slot = std::make_unique<Slot>(romPath, dataPath, scenarioPath, m_initialState, i);
			if (i == 0) {
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
			}
			slot->frameStack.resize(static_cast<size_t>(m_frameStack) * m_singleObsSize);
			slot->lastMask.assign(static_cast<size_t>(m_numButtons), 0);
			m_slots.emplace_back(std::move(slot));
		}
	}

	py::tuple reset(py::object seedObj = py::none()) {
		if (!seedObj.is_none()) {
			const uint64_t seed = py::int_(seedObj);
			for (size_t i = 0; i < m_slots.size(); ++i) {
				m_slots[i]->rng.seed(static_cast<uint32_t>(seed + i));
			}
		}
		py::array_t<uint8_t> obs(py::array::ShapeContainer{
			static_cast<py::ssize_t>(m_slots.size()),
			static_cast<py::ssize_t>(m_obsHeight),
			static_cast<py::ssize_t>(m_obsWidth),
			static_cast<py::ssize_t>(m_stackedChannels),
		});
		uint8_t* obsData = obs.mutable_data();
		std::vector<std::string> errors(m_slots.size());
		{
			py::gil_scoped_release release;
			batchThreadPool(m_numThreads).parallelFor(m_slots.size(), [&](size_t index) {
				try {
					resetSlot(*m_slots[index], obsData + index * m_stackedObsSize);
				} catch (const std::exception& exc) {
					errors[index] = exc.what();
				} catch (...) {
					errors[index] = "unknown native vector reset error";
				}
			});
		}
		throwFirstError(errors);
		py::list infos;
		for (size_t i = 0; i < m_slots.size(); ++i) {
			infos.append(py::dict());
		}
		return py::make_tuple(obs, infos);
	}

	py::tuple step(py::array_t<uint8_t> masks) {
		auto mask = masks.unchecked<2>();
		if (static_cast<size_t>(mask.shape(0)) != m_slots.size()) {
			throw std::runtime_error("actions first dimension must match num_envs");
		}
		if (mask.shape(1) != m_numButtons) {
			throw std::runtime_error("actions second dimension must match num_buttons");
		}
		py::array_t<uint8_t> obs(py::array::ShapeContainer{
			static_cast<py::ssize_t>(m_slots.size()),
			static_cast<py::ssize_t>(m_obsHeight),
			static_cast<py::ssize_t>(m_obsWidth),
			static_cast<py::ssize_t>(m_stackedChannels),
		});
		py::array_t<float> rewardArray({ static_cast<py::ssize_t>(m_slots.size()) });
		py::array_t<uint8_t> doneArray({ static_cast<py::ssize_t>(m_slots.size()) });
		uint8_t* obsData = obs.mutable_data();
		auto rewards = rewardArray.mutable_unchecked<1>();
		auto dones = doneArray.mutable_unchecked<1>();
		std::vector<std::string> errors(m_slots.size());
		std::vector<std::vector<uint8_t>> terminalObservations(m_slots.size());
		{
			py::gil_scoped_release release;
			batchThreadPool(m_numThreads).parallelFor(m_slots.size(), [&](size_t index) {
				try {
					StepOutput output;
					std::vector<uint8_t> action(static_cast<size_t>(m_numButtons));
					for (int key = 0; key < m_numButtons; ++key) {
						action[static_cast<size_t>(key)] = mask(static_cast<py::ssize_t>(index), key) ? 1 : 0;
					}
					stepSlot(*m_slots[index], action, obsData + index * m_stackedObsSize, output);
					rewards(static_cast<py::ssize_t>(index)) = output.reward;
					dones(static_cast<py::ssize_t>(index)) = output.done ? 1 : 0;
					if (output.done) {
						terminalObservations[index] = std::move(output.terminalObservation);
					}
				} catch (const std::exception& exc) {
					errors[index] = exc.what();
				} catch (...) {
					errors[index] = "unknown native vector step error";
				}
			});
		}
		throwFirstError(errors);
		py::list infos;
		for (size_t i = 0; i < m_slots.size(); ++i) {
			py::dict info = m_slots[i]->data.lookupAll();
			if (!terminalObservations[i].empty()) {
				py::array_t<uint8_t> terminal(py::array::ShapeContainer{
					static_cast<py::ssize_t>(m_obsHeight),
					static_cast<py::ssize_t>(m_obsWidth),
					static_cast<py::ssize_t>(m_stackedChannels),
				});
				std::memcpy(terminal.mutable_data(), terminalObservations[i].data(), m_stackedObsSize);
				info["terminal_observation"] = terminal;
				info["reset_info"] = py::dict();
				info["TimeLimit.truncated"] = false;
			}
			infos.append(info);
		}
		return py::make_tuple(obs, rewardArray, doneArray, infos);
	}

	py::tuple observationShape() const {
		return py::make_tuple(m_obsHeight, m_obsWidth, m_stackedChannels);
	}

	size_t numEnvs() const {
		return m_slots.size();
	}

private:
	struct Slot {
		Slot(const string& romPath, const string& dataPath, const string& scenarioPath, const std::string& initialState, size_t index)
			: emulator(std::make_unique<PyRetroEmulator>(romPath))
			, rng(static_cast<uint32_t>(0xC0D3u + index * 9973u)) {
			emulator->configureData(data);
			ScriptContext::reset();
			if (!data.m_data.load(dataPath) || !data.m_scen.load(scenarioPath)) {
				throw std::runtime_error("failed to load data or scenario");
			}
			if (!initialState.empty() && !emulator->m_re.unserialize(initialState.data(), initialState.size())) {
				throw std::runtime_error("failed to load initial state");
			}
			emulator->m_re.run();
			data.reset();
			data.updateRam();
		}

		std::unique_ptr<PyRetroEmulator> emulator;
		PyGameData data;
		std::vector<uint8_t> frameStack;
		std::vector<uint8_t> lastMask;
		bool hasLastMask = false;
		std::mt19937 rng;
	};

	struct StepOutput {
		float reward = 0.0f;
		bool done = false;
		std::vector<uint8_t> terminalObservation;
	};

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

	void readProcessedFrame(Slot& slot, std::vector<uint8_t>& singleObs) {
		std::vector<uint8_t> rgb;
		slot.emulator->readRgbFrame(rgb);
		singleObs.resize(m_singleObsSize);
		processRgbFrameToBuffer(
			rgb,
			slot.emulator->m_re.getImageWidth(),
			slot.emulator->m_re.getImageHeight(),
			m_crop,
			m_resize,
			m_grayscale,
			m_algorithm,
			singleObs.data()
		);
	}

	void resetFrameStack(Slot& slot, const std::vector<uint8_t>& singleObs) {
		for (int frame = 0; frame < m_frameStack; ++frame) {
			std::memcpy(slot.frameStack.data() + static_cast<size_t>(frame) * m_singleObsSize, singleObs.data(), m_singleObsSize);
		}
	}

	void pushFrame(Slot& slot, const std::vector<uint8_t>& singleObs) {
		if (m_frameStack == 1) {
			std::memcpy(slot.frameStack.data(), singleObs.data(), m_singleObsSize);
			return;
		}
		std::memmove(slot.frameStack.data(), slot.frameStack.data() + m_singleObsSize, static_cast<size_t>(m_frameStack - 1) * m_singleObsSize);
		std::memcpy(slot.frameStack.data() + static_cast<size_t>(m_frameStack - 1) * m_singleObsSize, singleObs.data(), m_singleObsSize);
	}

	void writeStackedObservation(const Slot& slot, uint8_t* dst) const {
		const size_t pixelCount = static_cast<size_t>(m_obsHeight) * static_cast<size_t>(m_obsWidth);
		for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
			for (int frame = 0; frame < m_frameStack; ++frame) {
				const uint8_t* src = slot.frameStack.data() + static_cast<size_t>(frame) * m_singleObsSize + pixel * static_cast<size_t>(m_obsChannels);
				uint8_t* out = dst + pixel * static_cast<size_t>(m_stackedChannels) + static_cast<size_t>(frame) * static_cast<size_t>(m_obsChannels);
				std::memcpy(out, src, static_cast<size_t>(m_obsChannels));
			}
		}
	}

	float clipReward(float reward) const {
		if (!m_rewardClip) {
			return reward;
		}
		return std::max(m_rewardClipLow, std::min(m_rewardClipHigh, reward));
	}

	void resetSlot(Slot& slot, uint8_t* dst) {
		if (!m_initialState.empty()) {
			if (!slot.emulator->m_re.unserialize(m_initialState.data(), m_initialState.size())) {
				throw std::runtime_error("failed to load initial state");
			}
		} else {
			slot.emulator->m_re.reset();
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
				slot.data.m_scen.update();
				if (slot.data.m_scen.isDone()) {
					break;
				}
			}
		}
		std::vector<uint8_t> singleObs;
		readProcessedFrame(slot, singleObs);
		resetFrameStack(slot, singleObs);
		writeStackedObservation(slot, dst);
	}

	void stepSlot(Slot& slot, const std::vector<uint8_t>& requestedAction, uint8_t* dst, StepOutput& output) {
		std::vector<uint8_t> action = requestedAction;
		if (slot.hasLastMask && m_stickyActionProb > 0.0) {
			std::uniform_real_distribution<double> stickyDist(0.0, 1.0);
			if (stickyDist(slot.rng) < m_stickyActionProb) {
				action = slot.lastMask;
			}
		}
		slot.lastMask = action;
		slot.hasLastMask = true;
		setKeys(slot, action);

		bool done = false;
		float totalReward = 0.0f;
		bool sawFrame = false;
		std::vector<uint8_t> prevRgb;
		std::vector<uint8_t> currRgb;
		for (int i = 0; i < m_frameSkip; ++i) {
			slot.emulator->m_re.run();
			slot.data.m_data.updateRam();
			slot.data.m_scen.update();
			totalReward += slot.data.m_scen.currentReward();
			done = slot.data.m_scen.isDone();
			if (m_maxpoolLastTwo && (i >= m_frameSkip - 2 || done)) {
				prevRgb.swap(currRgb);
				slot.emulator->readRgbFrame(currRgb);
				sawFrame = true;
			}
			if (done) {
				break;
			}
		}

		std::vector<uint8_t> singleObs(m_singleObsSize);
		if (!m_maxpoolLastTwo || !sawFrame) {
			readProcessedFrame(slot, singleObs);
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
				m_resize,
				m_grayscale,
				m_algorithm,
				singleObs.data()
			);
		}
		pushFrame(slot, singleObs);
		output.reward = clipReward(totalReward);
		output.done = done;
		if (done) {
			output.terminalObservation.resize(m_stackedObsSize);
			writeStackedObservation(slot, output.terminalObservation.data());
			resetSlot(slot, dst);
		} else {
			writeStackedObservation(slot, dst);
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
	std::string m_initialState;
	int m_numButtons = 0;
	int m_frameSkip = 1;
	int m_frameStack = 1;
	NativeCrop m_crop;
	NativeResize m_resize;
	bool m_grayscale = false;
	string m_algorithm;
	bool m_maxpoolLastTwo = false;
	int m_noopResetMax = 0;
	double m_stickyActionProb = 0.0;
	bool m_filterActions = false;
	bool m_rewardClip = false;
	float m_rewardClipLow = -1.0f;
	float m_rewardClipHigh = 1.0f;
	int m_numThreads = 1;
	long m_obsHeight = 0;
	long m_obsWidth = 0;
	int m_obsChannels = 0;
	int m_stackedChannels = 0;
	size_t m_singleObsSize = 0;
	size_t m_stackedObsSize = 0;
};

	py::tuple PyRetroEmulator::stepRepeatAndProcess(PyGameData& data, py::array_t<uint8_t> mask, int repeats, py::object cropObj, py::object resizeObj, bool grayscale, const string& algorithm, bool maxpoolLastTwo) {
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

	py::array_t<uint8_t> obs = processRgbFrame(currRgb, cropObj, resizeObj, grayscale, algorithm);
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
		.def("get_processed_screen", &PyRetroEmulator::getProcessedScreen)
		.def("step_repeat_and_process", &PyRetroEmulator::stepRepeatAndProcess)
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

	py::class_<PyNativeVectorEnv>(m, "NativeVectorEnv")
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
			py::arg("num_threads") = 0
		)
		.def("reset", &PyNativeVectorEnv::reset, py::arg("seed") = py::none())
		.def("step", &PyNativeVectorEnv::step)
		.def("observation_shape", &PyNativeVectorEnv::observationShape)
		.def_property_readonly("num_envs", &PyNativeVectorEnv::numEnvs);

	m.def("core_path", &::corePath, py::arg("hint") = py::none());
	m.def("data_path", &::dataPath, py::arg("hint") = py::none());
	}
