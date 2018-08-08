#include "imagecv.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>


namespace colorspaces {
  Image::Format::Format(const std::string name, const int id, const int cvType, imageCtor ctor, imageCvt cvt)
    : name(name), id(id), cvType(cvType),ctor(ctor),cvt(cvt) {}

  std::vector<Image::FormatPtr>& Image::Format::formatTable(){
    static std::vector<FormatPtr> formatTable;
    return formatTable;
  }

  const Image::FormatPtr Image::Format::searchFormat(const std::string name){
    std::vector<FormatPtr>::iterator it;

    for (it = formatTable().begin(); it != formatTable().end(); it++){
      if (name.compare((*it)->name) == 0)
	return *it;
    }
    return FormatPtr();
  }

  const Image::FormatPtr  Image::Format::createFormat(const std::string name, const int cvType, imageCtor ctor, imageCvt cvt){
    int id = formatTable().size();
    FormatPtr nFmt(new Format(name,id,cvType,ctor,cvt));
    formatTable().push_back(nFmt);
    return nFmt;
  }

  Image* Image::Format::createInstance(const int width, const int height, void *const data){
    if (ctor)
      return ctor(width,height,data);
    return 0;
  }

  Image& Image::convert(Image& dst) const throw(NoConversion){
    //std::cerr << "colorspaces: imagecv: convert: " << *_format << "->" << *dst._format << std::endl;
    return _format->cvt(*this,dst);
  }

  Image Image::clone() const{
    Image copy(cv::Mat::clone(),_format);
    return copy;
  }


  //static definitions
  const Image::FormatPtr Image::FORMAT_NONE 			= Image::Format::createFormat("NONE",0,0,0);
  const Image::FormatPtr ImageRGB8::FORMAT_RGB8 		= Image::Format::createFormat("RGB8",CV_8UC3,&ImageRGB8::createInstance,&ImageRGB8::imageCvt);
  const Image::FormatPtr ImageRGB8::FORMAT_RGB8_Z 		= Image::Format::createFormat("RGB8_Z",CV_8UC3,&ImageRGB8::createInstance,&ImageRGB8::imageCvt);
  const Image::FormatPtr ImageRGB8::FORMAT_DEPTH8_16 	= Image::Format::createFormat("DEPTH8_16",CV_8UC3,&ImageRGB8::createInstance,&ImageRGB8::imageCvt);
  const Image::FormatPtr ImageRGB8::FORMAT_DEPTH8_16_Z	= Image::Format::createFormat("DEPTH8_16_Z",CV_8UC3,&ImageRGB8::createInstance,&ImageRGB8::imageCvt);

  const Image::FormatPtr ImageYUY2::FORMAT_YUY2 = Image::Format::createFormat("YUY2",CV_8UC2,&ImageYUY2::createInstance,&ImageYUY2::imageCvt);
  const Image::FormatPtr ImageGRAY8::FORMAT_GRAY8 = Image::Format::createFormat("GRAY8",CV_8UC1,&ImageGRAY8::createInstance,&ImageGRAY8::imageCvt);
  const Image::FormatPtr ImageGRAY8::FORMAT_GRAY8_Z = Image::Format::createFormat("GRAY8_Z",CV_8UC1,&ImageGRAY8::createInstance,&ImageGRAY8::imageCvt);
  const Image::FormatPtr ImageHSV8::FORMAT_HSV8 = Image::Format::createFormat("HSV8",CV_8UC3,&ImageHSV8::createInstance,&ImageHSV8::imageCvt);
  const Image::FormatPtr ImageYCRCB::FORMAT_YCRCB = Image::Format::createFormat("YCRCB",CV_8UC3,&ImageYCRCB::createInstance,&ImageYCRCB::imageCvt);
  const Image::FormatPtr ImageNV21::FORMAT_NV21 = Image::Format::createFormat("NV21",CV_8UC2,&ImageNV21::createInstance,&ImageNV21::imageCvt);


  Image::Image()
    : cv::Mat(),width(0),height(0),_format() {}

  Image::Image(const int width, const int height, const FormatPtr fmt)
    : cv::Mat(height,width,fmt->cvType),width(width),height(height),_format(fmt) {}

  Image::Image(const int width, const int height, const FormatPtr fmt, void *const data)
    : cv::Mat(height,width,fmt->cvType,data),width(width),height(height),_format(fmt) {}

  Image::Image(const Image& i)
    : cv::Mat(i),width(i.width),height(i.height),_format(i.format()) {}

  Image::Image(const cv::Mat& m, const FormatPtr fmt)
    : cv::Mat(m),width(m.cols),height(m.rows),_format(fmt) {}

  ImageRGB8::ImageRGB8(const int width, const int height)
    : Image(width,height,FORMAT_RGB8) {}

  ImageRGB8::ImageRGB8(const int width, const int height, void *const data)
    : Image(width,height,FORMAT_RGB8,data) {}

  ImageRGB8::ImageRGB8(const Image& i)
    : Image(i.width,i.height,FORMAT_RGB8) {
    i.convert(*this);
  }

  ImageRGB8 ImageRGB8::read(const std::string& filename){
    struct stat s;
    if (stat(filename.c_str(),&s) == -1)
      throw std::runtime_error(filename+" not found");
    cv::Mat readImage(cv::imread(filename));//BGR
    cv::cvtColor(readImage,readImage,cv::COLOR_BGR2RGB);
    return ImageRGB8(Image(readImage,FORMAT_RGB8));
  }

  bool ImageRGB8::write(const std::string& filename,const std::vector<int>& params){
    cv::Mat bgrImage(this->size(),this->type());
    cv::cvtColor(*this,bgrImage,cv::COLOR_RGB2BGR);
    return cv::imwrite(filename, bgrImage, params);
  }

  Image& ImageRGB8::imageCvt(const Image& src, Image& dst) throw(NoConversion){
    assert((src.format() == FORMAT_RGB8 || src.format() == FORMAT_RGB8_Z
    		|| src.format() == FORMAT_DEPTH8_16 || src.format() == FORMAT_DEPTH8_16_Z) && "src is not a RGB8 image");


    if (dst.format() == FORMAT_RGB8 || src.format() == FORMAT_DEPTH8_16 || src.format() == FORMAT_RGB8_Z || src.format() == FORMAT_DEPTH8_16_Z)
      dst = src;
    else {
      const ImageRGB8 srcRgb8(src);//cast src to rgb image
      if (dst.format() == ImageYUY2::FORMAT_YUY2)
	srcRgb8.toYUY2(dst);
      else if (dst.format() == ImageGRAY8::FORMAT_GRAY8)
	srcRgb8.toGRAY8(dst);
      else if (dst.format() == ImageYCRCB::FORMAT_YCRCB)
	srcRgb8.toYCRCB(dst);
      else if (dst.format() == ImageHSV8::FORMAT_HSV8)
	srcRgb8.toHSV8(dst);
      else
	throw Image::NoConversion();
    }
    return dst;
  }

  void ImageRGB8::toGRAY8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageGRAY8::FORMAT_GRAY8)
      throw Image::FormatMismatch("FORMAT_GRAY8 required for dst");
    cv::cvtColor(*this,dst,cv::COLOR_RGB2GRAY);
  }

  void ImageRGB8::toYUY2(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageYUY2::FORMAT_YUY2)
      throw Image::FormatMismatch("FORMAT_YUY2 required for dst");
    if ((dst.width % 2 != 0) || (this->width % 2 != 0))
      throw Image::FormatMismatch("src and dst images have to have even number of columns");

    cv::Mat_<cv::Vec3b> ycrcb(dst.height,dst.width,dst.type());//YUV444 previous conversion
    cv::Mat_<cv::Vec2b> yuy2(dst);
    cv::cvtColor(*this,ycrcb,cv::COLOR_RGB2YCrCb);

    for (int i=0; i < height; i++){
      for (int j=0; j < width; j+=2){//two pixels each loop
	yuy2(i,j)[0] = ycrcb(i,j)[0];//Y0
	yuy2(i,j)[1] = ycrcb(i,j)[2];//U0
	yuy2(i,j+1)[0] = ycrcb(i,j+1)[0];//Y1
	yuy2(i,j+1)[1] = ycrcb(i,j)[1];//V0
      }
    }
  }

  void ImageRGB8::toHSV8(Image& dst) const throw(FormatMismatch){
    if (dst.format() != ImageHSV8::FORMAT_HSV8)
      throw Image::FormatMismatch("FORMAT_HSV8 required for dst");
    cv::cvtColor(*this,dst,cv::COLOR_RGB2HSV);
  }

  void ImageRGB8::toYCRCB(Image& dst) const throw(FormatMismatch){
    if (dst.format() != ImageYCRCB::FORMAT_YCRCB)
      throw Image::FormatMismatch("FORMAT_YCRCB required for dst");
    cv::cvtColor(*this,dst,cv::COLOR_RGB2YCrCb);
  }

  Image* ImageRGB8::createInstance(const int width, const int height, void *const data){
    if (data)
      return new ImageRGB8(width,height,data);
    else
      return new ImageRGB8(width,height);
  }

  ImageYUY2::ImageYUY2(const int width, const int height)
    : Image(width,height,FORMAT_YUY2) {}

  ImageYUY2::ImageYUY2(const int width, const int height, void *const data)
    : Image(width,height,FORMAT_YUY2,data) {}

  ImageYUY2::ImageYUY2(const Image& i)
    : Image(i.width,i.height,FORMAT_YUY2) {
    i.convert(*this);
  }

  Image& ImageYUY2::imageCvt(const Image& src, Image& dst) throw(NoConversion){
    assert(src.format() == FORMAT_YUY2 && "src is not a YUY2 image");
    if (dst.format() == FORMAT_YUY2)
      dst = src;
    else {
      const ImageYUY2 srcYuy2(src);
      if (dst.format() == ImageRGB8::FORMAT_RGB8)
	srcYuy2.toRGB8(dst);
      else if (dst.format() == ImageGRAY8::FORMAT_GRAY8)
	srcYuy2.toGRAY8(dst);
      else if (dst.format() == ImageYCRCB::FORMAT_YCRCB)
	srcYuy2.toYCRCB(dst);
      else
	throw Image::NoConversion();
    }
    return dst;
  }

  void ImageYUY2::toGRAY8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageGRAY8::FORMAT_GRAY8)
      throw Image::FormatMismatch("FORMAT_GRAY8 required for dst");

    int fromTo[] = {0,0};//first channel of YUY2 have the luminance information
    cv::mixChannels(this,1,&dst,1,fromTo,1);
  }

  void ImageYUY2::toRGB8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageRGB8::FORMAT_RGB8)
      throw Image::FormatMismatch("FORMAT_RGB8 required for dst");
    if ((dst.width % 2 != 0) || (this->width % 2 != 0))
      throw Image::FormatMismatch("src and dst images have to have even number of columns");

    ImageYCRCB ycrcbImg(dst.height,dst.width);//YCRCB previous conversion
    toYCRCB(ycrcbImg);

    cv::cvtColor(ycrcbImg,dst,cv::COLOR_YCrCb2RGB);
  }

  void ImageYUY2::toYCRCB(Image& dst) const throw(FormatMismatch){
    if (dst.format() != ImageYCRCB::FORMAT_YCRCB)
      throw Image::FormatMismatch("FORMAT_YCRCB required for dst");
    cv::Mat_<cv::Vec3b> ycrcb(dst);
    cv::Mat_<cv::Vec2b> yuy2(*this);

    for (int i=0; i < height; i++){
      for (int j=0; j < width; j+=2){//two pixels each loop
	ycrcb(i,j)[0] = yuy2(i,j)[0];//Y0<-Y0
	ycrcb(i,j)[1] = yuy2(i,j+1)[1];//V0<-V0
	ycrcb(i,j)[2] = yuy2(i,j)[1];//U0<-U0
	ycrcb(i,j+1)[0] = yuy2(i,j+1)[0];//Y1<-Y1
	ycrcb(i,j+1)[1] = yuy2(i,j+1)[1];//V1<-V0
	ycrcb(i,j+1)[2] = yuy2(i,j)[1];//U1<-U0
      }
    }
  }

  Image* ImageYUY2::createInstance(const int width, const int height, void *const data){
    if (data)
      return new ImageYUY2(width,height,data);
    else
      return new ImageYUY2(width,height);
  }

  ImageGRAY8::ImageGRAY8(const int width, const int height)
    : Image(width,height,FORMAT_GRAY8) {}

  ImageGRAY8::ImageGRAY8(const int width, const int height, void *const data)
    : Image(width,height,FORMAT_GRAY8,data) {}

  ImageGRAY8::ImageGRAY8(const Image& i)
    : Image(i.width,i.height,FORMAT_GRAY8) {
    i.convert(*this);
  }

  ImageGRAY8 ImageGRAY8::read(const std::string& filename){
    struct stat s;
    if (stat(filename.c_str(),&s) == -1)
      throw std::runtime_error(filename+" not found");
    return ImageGRAY8(Image(cv::imread(filename,0),FORMAT_GRAY8));
  }

  bool ImageGRAY8::write(const std::string& filename,const std::vector<int>& params){
    return cv::imwrite(filename, *this, params);
  }

  Image& ImageGRAY8::imageCvt(const Image& src, Image& dst) throw(NoConversion){
    assert(src.format() == FORMAT_GRAY8 && "src is not a GRAY8 image");
    if (dst.format() == FORMAT_GRAY8)
      dst = src;
    else {
      const ImageGRAY8 srcGray8(src);
      if (dst.format() == ImageYUY2::FORMAT_YUY2)
	srcGray8.toYUY2(dst);
      else if (dst.format() == ImageRGB8::FORMAT_RGB8)
	srcGray8.toRGB8(dst);
      else
	throw Image::NoConversion();
    }
    return dst;
  }

  void ImageGRAY8::toRGB8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageRGB8::FORMAT_RGB8)
      throw Image::FormatMismatch("FORMAT_RGB8 required for dst");

    cv::cvtColor(*this,dst,cv::COLOR_GRAY2RGB);
  }

  void ImageGRAY8::toYUY2(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageYUY2::FORMAT_YUY2)
      throw Image::FormatMismatch("FORMAT_YUY2 required for dst");
    //U and V will be 0
    cv::Mat uv(cv::Mat::zeros(width,height,FORMAT_GRAY8->cvType));

    int fromTo[] = {0,0 , 1,1};//GRAY to Y channel, 0->U/V
    cv::Mat src[] = {*this,uv};
    cv::mixChannels(src,2,&dst,1,fromTo,1);
  }

  Image* ImageGRAY8::createInstance(const int width, const int height, void *const data){
    if (data)
      return new ImageGRAY8(width,height,data);
    else
      return new ImageGRAY8(width,height);
  }


  ImageHSV8::ImageHSV8(const int width, const int height)
    : Image(width,height,FORMAT_HSV8) {}

  ImageHSV8::ImageHSV8(const int width, const int height, void *const data)
    : Image(width,height,FORMAT_HSV8,data) {}

  ImageHSV8::ImageHSV8(const Image& i)
    : Image(i.width,i.height,FORMAT_HSV8) {
    i.convert(*this);
  }

  Image& ImageHSV8::imageCvt(const Image& src, Image& dst) throw(NoConversion){
    assert(src.format() == FORMAT_HSV8 && "src is not a HSV8 image");
    if (dst.format() == FORMAT_HSV8)
      dst = src;
    else {
      const ImageHSV8 srcHsv8(src);
      if (dst.format() == ImageRGB8::FORMAT_RGB8)
	srcHsv8.toRGB8(dst);
      else
	throw Image::NoConversion();
    }
    return dst;
  }

  void ImageHSV8::toRGB8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageRGB8::FORMAT_RGB8)
      throw Image::FormatMismatch("FORMAT_RGB8 required for dst");

    cv::cvtColor(*this,dst,cv::COLOR_HSV2RGB);
  }

  Image* ImageHSV8::createInstance(const int width, const int height, void *const data){
    if (data)
      return new ImageHSV8(width,height,data);
    else
      return new ImageHSV8(width,height);
  }

  ImageYCRCB::ImageYCRCB(const int width, const int height)
    : Image(width,height,FORMAT_YCRCB) {}

  ImageYCRCB::ImageYCRCB(const int width, const int height, void *const data)
    : Image(width,height,FORMAT_YCRCB,data) {}

  ImageYCRCB::ImageYCRCB(const Image& i)
    : Image(i.width,i.height,FORMAT_YCRCB) {
    i.convert(*this);
  }

  Image& ImageYCRCB::imageCvt(const Image& src, Image& dst) throw(NoConversion){
    assert(src.format() == FORMAT_YCRCB && "src is not a YCRCB image");
    if (dst.format() == FORMAT_YCRCB)
      dst = src;
    else {
      const ImageYCRCB srcYcrcb(src);
      if (dst.format() == ImageRGB8::FORMAT_RGB8)
	srcYcrcb.toRGB8(dst);
      else
	throw Image::NoConversion();
    }
    return dst;
  }

  void ImageYCRCB::toRGB8(Image& dst) const throw(FormatMismatch){
    if (dst.format() != ImageRGB8::FORMAT_RGB8)
      throw Image::FormatMismatch("FORMAT_RGB8 required for dst");

    cv::cvtColor(*this,dst,cv::COLOR_YCrCb2RGB);
  }

  Image* ImageYCRCB::createInstance(const int width, const int height, void *const data){
    if (data)
      return new ImageYCRCB(width,height,data);
    else
      return new ImageYCRCB(width,height);
  }

  ImageNV21::ImageNV21(const int width, const int height)
    : Image(width,height,FORMAT_NV21) {}

  ImageNV21::ImageNV21(const int width, const int height, void *const data)
    : Image(width,height,FORMAT_NV21,data) {}

  ImageNV21::ImageNV21(const Image& i)
    : Image(i.width,i.height,FORMAT_NV21) {
    i.convert(*this);
  }

  Image& ImageNV21::imageCvt(const Image& src, Image& dst) throw(NoConversion){
    assert(src.format() == FORMAT_NV21 && "src is not a NV21 image");
    if (dst.format() == FORMAT_NV21)
      dst = src;
    else {
      const ImageNV21 srcNv21(src);
      if (dst.format() == ImageRGB8::FORMAT_RGB8)
	srcNv21.toRGB8(dst);/*
      else if (dst.format() == ImageGRAY8::FORMAT_GRAY8)
	srcNv21.toGRAY8(dst);
      else if (dst.format() == ImageYCRCB::FORMAT_YCRCB)
	srcNv21.toYCRCB(dst);*/
      else
	throw Image::NoConversion();
    }
    return dst;
  }

  void ImageNV21::toGRAY8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageGRAY8::FORMAT_GRAY8)
      throw Image::FormatMismatch("FORMAT_GRAY8 required for dst");

    int fromTo[] = {0,0};//first channel of YUY2 have the luminance information (need to test!!)
    cv::mixChannels(this,1,&dst,1,fromTo,1);
  }

  void ImageNV21::toRGB8(Image& dst) const throw(Image::FormatMismatch){
    if (dst.format() != ImageRGB8::FORMAT_RGB8)
      throw Image::FormatMismatch("FORMAT_RGB8 required for dst");
    if ((dst.width % 2 != 0) || (this->width % 2 != 0))
      throw Image::FormatMismatch("src and dst images have to have even number of columns");

    //cv::cvtColor(*this,dst,cv::COLOR_YUV420sp2RGB);

    unsigned char *rgb = (unsigned char *)dst.data;
    unsigned char *yuv = (unsigned char *)this->data,
    	*yuv_y = yuv, *yuv_uv = yuv + dst.width * dst.height;
    for (int i = 0; i < height; i++, yuv_uv -= (i&1)?width:0) {
        for (int j = 0; j < width; j++, yuv_uv += (j&1)?0:2) {
            int y = *yuv_y++;
            y = (y < 16) ? 16 : y;
            int v = yuv_uv[0] - 128;
            int u = yuv_uv[1] - 128;

			int multi = 1.164f * (y - 16);
            int r = (int) (multi + 1.596f * v);
            int g = (int) (multi - 0.813f * v - 0.391f * u);
            int b = (int) (multi + 2.018f * u);

            *rgb++ = r < 0 ? 0 : (r > 255 ? 255 : r);
            *rgb++ = g < 0 ? 0 : (g > 255 ? 255 : g);
            *rgb++ = b < 0 ? 0 : (b > 255 ? 255 : b);
        }
    }
  }

  void ImageNV21::toYCRCB(Image& dst) const throw(FormatMismatch){
    if (dst.format() != ImageYCRCB::FORMAT_YCRCB)
      throw Image::FormatMismatch("FORMAT_YCRCB required for dst");
    ImageYCRCB rgbImg(dst.height,dst.width);
    toRGB8(rgbImg);
    cv::cvtColor(rgbImg,dst,cv::COLOR_RGB2YCrCb);
  }

  Image* ImageNV21::createInstance(const int width, const int height, void *const data){
    if (data)
      return new ImageNV21(width,height,data);
    else
      return new ImageNV21(width,height);
  }



//   ImageppPtr Imagepp::createTestHline(const int width,
// 				      const int height,
// 				      const int lineWidth,
// 				      const int startRow,
// 				      const RGBColor *bgColor,
// 				      const RGBColor *fgColor){
//     Image *i = Image_createTestHline(width,height,lineWidth,startRow,bgColor,fgColor);
//     if (i!=0){
//       ImageppPtr ii(new Imagepp(i->description,i->imageData));
//       Image_swapDataOwner(i,ii.get());
//       delete_Image(i);
//       return ii;
//     }else
//       return ImageppPtr();
//   }

//   ImageppPtr Imagepp::createTestVline(const int width,
// 				      const int height,
// 				      const int lineWidth,
// 				      const int startCol,
// 				      const RGBColor *bgColor,
// 				      const RGBColor *fgColor){
//     Image *i = Image_createTestVline(width,height,lineWidth,startCol,bgColor,fgColor);
//     if (i!=0){
//       ImageppPtr ii(new Imagepp(i->description,i->imageData));
//       Image_swapDataOwner(i,ii.get());
//       delete_Image(i);
//       return ii;
//     }else
//       return ImageppPtr();
//   }

//   ImageppPtr Imagepp::createTestSquare(const int width,
// 				       const int height,
// 				       const int sideLength,
// 				       const int xStartCorner,
// 				       const int yStartCorner,
// 				       const RGBColor *bgColor,
// 				       const RGBColor *fgColor){
//     Image *i = Image_createTestSquare(width,height,
// 				      sideLength,
// 				      xStartCorner,yStartCorner,
// 				      bgColor,bgColor);
//     if (i!=0){
//       ImageppPtr ii(new Imagepp(i->description,i->imageData));
//       Image_swapDataOwner(i,ii.get());
//       delete_Image(i);
//       return ii;
//     }else
//       return ImageppPtr();
//   }


}//namespace

/**
 * Insert a format in an output stream. Only debugging, output could be truncated
 */
std::ostream &operator<<(std::ostream &stream, const colorspaces::Image::Format& fmt){
  stream << "FMT("<< fmt.name << ";channels:" << CV_MAT_CN(fmt.cvType) << ";depth:" << CV_MAT_DEPTH(fmt.cvType) << ")";
  return stream;
}

std::ostream &operator<<(std::ostream &stream, const colorspaces::Image& img){
  stream << "IMG(" << *(img.format()) << ";" << img.width << "x" << img.height << ")";
  return stream;
}
