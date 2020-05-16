/*
 *
 *  Copyright (C) 1997-2009 JDERobot Developers Team
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 *  Authors : David Lobato Bravo <dav.lobato@gmail.com>
 *
 */

#ifndef IMAGECV_COLORSPACES_H
#define IMAGECV_COLORSPACES_H

#include <opencv2/core/core.hpp>
#include <exception>
#include <vector>
#include <iostream>
#include <memory>
#include "uncopyable.h"

namespace colorspaces {
  /**
   * An image
   */
  class Image: public  cv::Mat {
  public:
    class FormatMismatch: public std::exception{
    public:
      FormatMismatch(const std::string msg)
	: message(msg) {}
      ~FormatMismatch() throw() {}
      virtual const char* what() const throw()
      {
	return message.c_str();
      }
    private:
      const std::string message;
    };

    class NoConversion: public std::exception{
      virtual const char* what() const throw()
      {
	return "Can't convert image to requested format";
      }
    };

    typedef Image* (*imageCtor)(const int height, const int width, void *const data);
    typedef Image& (*imageCvt)(const Image& src, Image& dst);

    class Format;
    typedef std::shared_ptr<Format> FormatPtr;

    /**
     * Define the pixel format of an image
     */
    class Format: public Uncopyable{
    public:
      static const FormatPtr createFormat(const std::string name, const int cvType, imageCtor ctor, imageCvt cvt);
      static const FormatPtr getFormat(const int fmtId);
      static const FormatPtr searchFormat(const std::string name);
      Image* createInstance(const int width, const int height, void *const data);
      int bytesPerPixel() { return CV_ELEM_SIZE(cvType); }
      std::string name;/**< String that represents this format*/
      int id;/**< Format identification*/
      int cvType;/**< Opencv data type used for pixel data*/
      imageCtor ctor;/**< Contructor*/
      imageCvt cvt;/**< conversion function*/
    private:
      Format(const std::string name, const int id, const int cvType, imageCtor ctor, imageCvt cvt);
      static std::vector<FormatPtr>& formatTable();
    };


    /**
     * Constructor
     */
    Image();

    /**
     * Constructor
     */
    Image(const int width, const int height, const FormatPtr fmt);

    /**
     * Constructor from user data
     */
    Image(const int width, const int height, const FormatPtr fmt, void *const data);

    /**
     * Copy onstructor
     */
    Image(const Image& i);

    /**
     * Copy onstructor from cv::Mat
     */
    Image(const cv::Mat& m, const FormatPtr fmt);

    /**
     * Image destructor
     */
    virtual ~Image() {}

    /**
     * Get image's format
     */
    const FormatPtr  format() const { return _format; }

    /**
     * Convert image to dst fmt
     */
    Image& convert(Image& dst) const throw(NoConversion);

    /**
     * Clone image returning a new copy
     */
    Image clone() const;

    int width;
    int height;

    static const FormatPtr FORMAT_NONE;
  private:
    FormatPtr _format;
  };


  /**
   * A RGB 24 bit image
   */
  class ImageRGB8: public Image {
  public:
    /**
     * Constructor
     */
    ImageRGB8(const int width, const int height);

    /**
     * Constructor from user data
     */
    ImageRGB8(const int width, const int height, void *const data);

    /**
     * Copy constructor from Image, conversion will happen if needed
     */
    ImageRGB8(const Image& i);

    /**
     * Conversion methods.
     */
    void toGRAY8(Image& dst) const throw(FormatMismatch);
    void toYUY2(Image& dst) const throw(FormatMismatch);
    void toHSV8(Image& dst) const throw(FormatMismatch);
    void toYCRCB(Image& dst) const throw(FormatMismatch);

    /**
     * Read from a file
     * See cv::imread for flags and params
     */
    static ImageRGB8 read(const std::string& filename);

    /**
     * Write to a file
     * See cv::imwrite for flags and params
     */
    bool write(const std::string& filename, const std::vector<int>& params=std::vector<int>());


    /**
     * Factory method
     */
    static Image* createInstance(const int width, const int height, void *const data);
    static Image& imageCvt(const Image& src, Image& dst) throw(NoConversion);
    static const FormatPtr FORMAT_RGB8, FORMAT_RGB8_Z, FORMAT_DEPTH8_16, FORMAT_DEPTH8_16_Z;
  };

  /**
   * A YUY2 image
   */
  class ImageYUY2: public  Image {
  public:
    /**
     * Constructor
     * Width have to be an even number.
     */
    ImageYUY2(const int width, const int height);

    /**
     * Constructor from user data
     * Width have to be an even number.
     */
    ImageYUY2(const int width, const int height, void *const data);

    /**
     * Copy constructor.
     * if \param i doesn't match format a conversion will happen.
     */
    ImageYUY2(const Image& i);


    /**
     * Conversion methods.
     * Returns a copy
     */
    void toGRAY8(Image& dst) const throw(FormatMismatch);
    void toRGB8(Image& dst) const throw(FormatMismatch);
    void toYCRCB(Image& dst) const throw(FormatMismatch);

    /**
     * Factory method
     */
    static Image* createInstance(const int width, const int height, void *const data);
    static Image& imageCvt(const Image& src, Image& dst) throw(NoConversion);
    static const FormatPtr FORMAT_YUY2;
  };


  /**
   * A Gray 8 bit image
   */
  class ImageGRAY8: public  Image {
  public:
    /**
     * Constructor
     */
    ImageGRAY8(const int width, const int height);

    /**
     * Constructor from user data
     */
    ImageGRAY8(const int width, const int height, void *const data);

    /**
     * Copy constructor.
     * if \param i doesn't match format a conversion will happen.
     */
    ImageGRAY8(const Image& i);


    /**
     * Conversion methods.
     * Returns a copy
     */
    void toRGB8(Image& dst) const throw(FormatMismatch);
    void toYUY2(Image& dst) const throw(FormatMismatch);

    /**
     * Read from a file
     * See cv::imread for flags and params
     */
    static ImageGRAY8 read(const std::string& filename);

    /**
     * Write to a file
     * See cv::imwrite for flags and params
     */
    bool write(const std::string& filename, const std::vector<int>& params=std::vector<int>());

    /**
     * Factory method
     */
    static Image* createInstance(const int width, const int height, void *const data);
    static Image& imageCvt(const Image& src, Image& dst) throw(NoConversion);
    static const FormatPtr FORMAT_GRAY8, FORMAT_GRAY8_Z;
  };

  /**
   * A HSV8 image
   */
  class ImageHSV8: public  Image {
  public:
    /**
     * Constructor
     */
    ImageHSV8(const int width, const int height);

    /**
     * Constructor from user data
     */
    ImageHSV8(const int width, const int height, void *const data);

    /**
     * Copy constructor.
     * if \param i doesn't match format a conversion will happen.
     */
    ImageHSV8(const Image& i);


    /**
     * Conversion methods.
     * Returns a copy
     */
    void toRGB8(Image& dst) const throw(FormatMismatch);

    /**
     * Factory method
     */
    static Image* createInstance(const int width, const int height, void *const data);
    static Image& imageCvt(const Image& src, Image& dst) throw(NoConversion);
    static const FormatPtr FORMAT_HSV8;
  };

  /**
   * A YCRCB image
   */
  class ImageYCRCB: public  Image {
  public:
    /**
     * Constructor
     */
    ImageYCRCB(const int width, const int height);

    /**
     * Constructor from user data
     */
    ImageYCRCB(const int width, const int height, void *const data);

    /**
     * Copy constructor.
     * if \param i doesn't match format a conversion will happen.
     */
    ImageYCRCB(const Image& i);


    /**
     * Conversion methods.
     * Returns a copy
     */
    void toRGB8(Image& dst) const throw(FormatMismatch);
    //void toYUY2(Image& dst) const throw(FormatMismatch);

    /**
     * Factory method
     */
    static Image* createInstance(const int width, const int height, void *const data);
    static Image& imageCvt(const Image& src, Image& dst) throw(NoConversion);
    static const FormatPtr FORMAT_YCRCB;
  };

  /**
   * A NV21 image
   */
  class ImageNV21: public  Image {
  public:
    /**
     * Constructor
     * Width have to be an even number.
     */
    ImageNV21(const int width, const int height);

    /**
     * Constructor from user data
     * Width have to be an even number.
     */
    ImageNV21(const int width, const int height, void *const data);

    /**
     * Copy constructor.
     * if \param i doesn't match format a conversion will happen.
     */
    ImageNV21(const Image& i);


    /**
     * Conversion methods.
     * Returns a copy
     */
    void toGRAY8(Image& dst) const throw(FormatMismatch);
    void toRGB8(Image& dst) const throw(FormatMismatch);
    void toYCRCB(Image& dst) const throw(FormatMismatch);

    /**
     * Factory method
     */
    static Image* createInstance(const int width, const int height, void *const data);
    static Image& imageCvt(const Image& src, Image& dst) throw(NoConversion);
    static const FormatPtr FORMAT_NV21;
  };


} //namespace

//declarations outside the namespace

/**
 * Insert a format in an output stream. Only debugging, output could be truncated
 */
std::ostream &operator<<(std::ostream &stream, const colorspaces::Image::Format& fmt);

/**
 * Insert an image in an output stream. Only debugging, output could be truncated
 */
std::ostream &operator<<(std::ostream &stream, const colorspaces::Image& img);

#endif //IMAGECV_COLORSPACES_H
