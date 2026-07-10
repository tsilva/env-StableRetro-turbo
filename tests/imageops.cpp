#include <cstdint>

#include <gmock/gmock.h>

#include "imageops.h"

using Retro::Image;
using testing::ElementsAreArray;

TEST(ImageOpsTest, Xrgb8888ConvertsToRgb888) {
	const uint32_t xrgb[] = {
		0x00C84848,
		0x00C66C3A,
	};
	uint8_t rgb[6] = {};
	const uint8_t expected[] = {
		0xC8, 0x48, 0x48,
		0xC6, 0x6C, 0x3A,
	};

	Image input(Image::Format::RGBX888, xrgb, 2, 1, sizeof(xrgb));
	Image output(Image::Format::RGB888, rgb, 2, 1, sizeof(rgb));
	input.copyTo(&output);

	EXPECT_THAT(rgb, ElementsAreArray(expected));
}
