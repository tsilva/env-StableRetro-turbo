#include "atari.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ale/vector/env_vectorizer.hpp"

namespace py = pybind11;
using ale::vector::Action;
using ale::vector::AutoresetMode;
using ale::vector::BatchResult;
using ale::vector::EnvVectorizer;

namespace {

template <typename T>
py::array_t<T> takeArray(T* data, const std::vector<py::ssize_t>& shape) {
	py::capsule owner(data, [](void* value) {
		delete[] static_cast<T*>(value);
	});
	return py::array_t<T>(shape, data, owner);
}

class PyAtariVecEnv {
public:
	PyAtariVecEnv(
		const std::string& romPath,
		int numEnvs,
		int batchSize,
		int numThreads,
		int threadAffinityOffset,
		int maxEpisodeSteps,
		float repeatActionProbability,
		bool fullActionSpace,
		const std::string& autoresetMode,
		int imageHeight,
		int imageWidth,
		bool grayscale,
		int stackNum,
		int frameSkip,
		bool maxpool,
		int noopMax,
		bool episodicLife,
		bool lifeLossInfo,
		bool rewardClipping,
		bool useFireReset
	)
		: m_grayscale(grayscale) {
		AutoresetMode mode;
		if (autoresetMode == "NextStep") {
			mode = AutoresetMode::NextStep;
		} else if (autoresetMode == "SameStep") {
			mode = AutoresetMode::SameStep;
		} else if (autoresetMode == "Disabled") {
			mode = AutoresetMode::Disabled;
		} else {
			throw std::invalid_argument("Invalid autoreset_mode: " + autoresetMode);
		}
		m_vectorizer = std::make_unique<EnvVectorizer>(
			romPath, numEnvs, batchSize, numThreads, threadAffinityOffset, mode,
			imageHeight, imageWidth, stackNum, grayscale, frameSkip, maxpool,
			noopMax, useFireReset, episodicLife, lifeLossInfo, rewardClipping,
			maxEpisodeSteps, repeatActionProbability, fullActionSpace
		);
	}

	py::tuple reset(const std::vector<int>& resetIndices, const std::vector<int>& resetSeeds) {
		BatchResult result = [&] {
			py::gil_scoped_release release;
			return m_vectorizer->reset(resetIndices, resetSeeds);
		}();
		return wrapReset(std::move(result));
	}

	py::tuple step(
		py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions
	) {
		send(std::move(actions));
		return recv();
	}

	void send(
		py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions
	) {
		if (actions.ndim() != 1 ||
			actions.shape(0) != static_cast<py::ssize_t>(m_vectorizer->batch_size())) {
			throw std::invalid_argument("actions must have shape (batch_size,)");
		}
		auto values = actions.unchecked<1>();
		std::vector<Action> nativeActions;
		nativeActions.reserve(static_cast<std::size_t>(values.shape(0)));
		for (py::ssize_t index = 0; index < values.shape(0); ++index) {
			Action action;
			action.env_id = static_cast<int>(index);
			action.action_id = static_cast<int>(values(index));
			action.paddle_strength = 1.0f;
			action.force_reset = false;
			action.snapshot_only = false;
			nativeActions.push_back(action);
		}

		py::gil_scoped_release release;
		m_vectorizer->send(nativeActions);
	}

	py::tuple recv() {
		BatchResult result = [&] {
			py::gil_scoped_release release;
			return m_vectorizer->recv();
		}();
		return wrapStep(std::move(result));
	}

	std::vector<int> actionSet() const {
		std::vector<int> result;
		const auto& actions = m_vectorizer->action_set();
		result.reserve(actions.size());
		for (const auto action : actions) {
			result.push_back(static_cast<int>(action));
		}
		return result;
	}

	int numEnvs() const { return m_vectorizer->num_envs(); }
	int batchSize() const { return m_vectorizer->batch_size(); }

private:
	std::vector<py::ssize_t> observationShape(std::size_t batchSize) const {
		auto [stackNum, height, width, channels] = m_vectorizer->observation_shape();
		std::vector<py::ssize_t> shape = {
			static_cast<py::ssize_t>(batchSize), stackNum, height, width
		};
		if (!m_grayscale) {
			shape.push_back(channels);
		}
		return shape;
	}

	py::tuple wrapReset(BatchResult&& result) {
		const std::size_t batchSize = result.batch_size();
		auto observations = takeArray(result.release_observations(), observationShape(batchSize));
		const std::vector<py::ssize_t> infoShape = {static_cast<py::ssize_t>(batchSize)};
		py::dict info;
		info["env_id"] = takeArray(result.release_env_ids(), infoShape);
		info["lives"] = takeArray(result.release_lives(), infoShape);
		info["frame_number"] = takeArray(result.release_frame_numbers(), infoShape);
		info["episode_frame_number"] =
			takeArray(result.release_episode_frame_numbers(), infoShape);
		return py::make_tuple(std::move(observations), std::move(info));
	}

	py::tuple wrapStep(BatchResult&& result) {
		const std::size_t batchSize = result.batch_size();
		const auto obsShape = observationShape(batchSize);
		const std::vector<py::ssize_t> infoShape = {static_cast<py::ssize_t>(batchSize)};
		bool anyDone = false;
		for (std::size_t index = 0; index < batchSize; ++index) {
			if (result.terminations_data()[index] || result.truncations_data()[index]) {
				anyDone = true;
				break;
			}
		}
		const bool hasFinalObservations = result.has_final_obs();

		auto observations = takeArray(result.release_observations(), obsShape);
		auto rewards = takeArray(result.release_rewards(), infoShape);
		auto terminations = takeArray(result.release_terminations(), infoShape);
		auto truncations = takeArray(result.release_truncations(), infoShape);
		py::dict info;
		info["env_id"] = takeArray(result.release_env_ids(), infoShape);
		info["lives"] = takeArray(result.release_lives(), infoShape);
		info["frame_number"] = takeArray(result.release_frame_numbers(), infoShape);
		info["episode_frame_number"] =
			takeArray(result.release_episode_frame_numbers(), infoShape);
		if (hasFinalObservations && anyDone) {
			info["final_obs"] = takeArray(result.release_final_observations(), obsShape);
		}
		return py::make_tuple(
			std::move(observations), std::move(rewards), std::move(terminations),
			std::move(truncations), std::move(info)
		);
	}

	bool m_grayscale;
	std::unique_ptr<EnvVectorizer> m_vectorizer;
};

}  // namespace

void bindAtariVecEnv(py::module_& module) {
	py::class_<PyAtariVecEnv>(module, "_AtariVecEnv")
		.def(
			py::init<
				const std::string&, int, int, int, int, int, float, bool,
				const std::string&, int, int, bool, int, int, bool, int,
				bool, bool, bool, bool>(),
			py::arg("rom_path"), py::arg("num_envs"), py::arg("batch_size") = 0,
			py::arg("num_threads") = 0, py::arg("thread_affinity_offset") = -1,
			py::arg("max_episode_steps") = 108000,
			py::arg("repeat_action_probability") = 0.0f,
			py::arg("full_action_space") = false,
			py::arg("autoreset_mode") = "NextStep", py::arg("img_height") = 84,
			py::arg("img_width") = 84, py::arg("grayscale") = true,
			py::arg("stack_num") = 4, py::arg("frameskip") = 4,
			py::arg("maxpool") = true, py::arg("noop_max") = 30,
			py::arg("episodic_life") = false, py::arg("life_loss_info") = false,
			py::arg("reward_clipping") = true, py::arg("use_fire_reset") = true
		)
		.def("reset", &PyAtariVecEnv::reset)
		.def("step", &PyAtariVecEnv::step)
		.def("send", &PyAtariVecEnv::send)
		.def("recv", &PyAtariVecEnv::recv)
		.def("get_action_set", &PyAtariVecEnv::actionSet)
		.def_property_readonly("num_envs", &PyAtariVecEnv::numEnvs)
		.def_property_readonly("batch_size", &PyAtariVecEnv::batchSize);
}
