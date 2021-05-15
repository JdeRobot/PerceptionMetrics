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
 *  Authors : Roberto Calvo Palomino <rocapal@gsyc.es>
 *            José María Cañas <jmplaza@gsyc.es>
 *            Pablo Miangolarra Tejada <pablo.miangolarra@gmail.com>
 *            David Lobato Bravo <dav.lobato@gmail.com>
 *
 */

#ifndef _COLOR_SPACES_H
#define _COLOR_SPACES_H

#define NAME     "colorspaces"
#define COLORSPACES_VERSION  "1.4.0"


#ifdef __cplusplus
extern "C" {
#endif

  /// *** RGB to HSI  *** ///

  struct HSV
  {
    double H;
    double S;
    double V;
  };
  
  extern struct HSV * LUT_RGB2HSV [64][64][64];

  extern int isInitTableHSV;
 
  /// \brief Init the RGB2HSV
  void RGB2HSV_init();

  /// \brief Create a translate RGB2HSV table with resolution of 6bits (64x64x64)
  void RGB2HSV_createTable();

  /// \brief Free de memory of RGB2HSV
  void RGB2HSV_destroyTable();

  /// \brief Print the struct HSV
  void RGB2HSV_printHSI (struct HSV*);

  /// \brief Test
  void RGB2HSV_test();

  /// \brief Returns the translation from RGB to HSV
  static inline const struct HSV* RGB2HSV_getHSV (int R, int G, int B)  { return LUT_RGB2HSV[R>>2][G>>2][B>>2]; }

  /// \brief Returns the translation from HSV to RGB
  void hsv2rgb(double H, double S, double V, double *r, double *g, double *b);


  /// *** RGB to YUV  *** ///

  struct YUV
  {
    double Y;
    double U;
    double V;
  };

  extern struct YUV * LUT_RGB2YUV [64][64][64];

  extern int isInitTableYUV;

  /// \brief Init the RGB2YUV
  void RGB2YUV_init();

  /// \brief Create a generic translate RGB2YUV table with resolution of 6bits (64x64x64)
  void RGB2YUV_createTable();

  /// \brief Free de memory of RGB2YUV
  void RGB2YUV_destroyTable();

  /// \brief Print the struct YUV
  void RGB2YUV_printYUV (struct YUV*);

  /// \brief Test
  void RGB2YUV_test();

  /// \brief Returns the translation from RGB to YUV
  static inline const struct YUV* RGB2YUV_getYUV (int R, int G, int B)  { return LUT_RGB2YUV[R>>2][G>>2][B>>2]; }

  /// \brief Returns the translation from YUV to RGB
  void yuv2rgb(double Y, double U, double V, double *r, double *g, double *b);


#ifdef __cplusplus
} /*extern "C"*/
#endif

#endif /*_COLOR_SPACES_H*/
