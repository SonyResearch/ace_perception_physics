// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <vector>
#include "half.hpp"

class ImageCHW {
 private:
  int channels_;
  int height_;
  int width_;
  std::vector<half_float::half> pixels_;

 public:
  /**
   * @brief Constructs an empty ImageCHW object.
   *
   */
  ImageCHW();

  /**
   * @brief Constructs an ImageCHW object.
   *
   * @param channels The number of image channels.
   * @param height The image height.
   * @param width The image width.
   */
  ImageCHW(int channels, int height, int width);

  /**
   * @brief Accesses a pixel in the image.
   *
   * @param c Channel index.
   * @param h Height index.
   * @param w Width index.
   * @return Reference to the pixel value.
   */
  half_float::half& At(int c, int h, int w);

  /**
   * @brief Resets all pixel values in the image to 0.
   */
  void ResetToZero();

  /**
   * @brief Returns a pointer to the underlying data array.
   *
   * @return Pointer to the underlying data array.
   */
  half_float::half* GetDataPointer();

  std::vector<half_float::half> GetVals();
};
