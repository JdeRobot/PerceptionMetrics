/*
 *
 *  Copyright (C) 1997-2010 JDE Developers Team
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
 *  Author : David Lobato Bravo <dav.lobato@gmail.com>
 *
 */


#ifndef CAMERA_ICE
#define CAMERA_ICE


#include <jderobot/image.ice>

module jderobot{
  /**
   * Static description of a camera
   */
  class CameraDescription
  {
    string name;
    string shortDescription;
    string streamingUri;
    float fdistx;
    float fdisty;
    float u0;
    float v0;
    float skew;
    float posx;
    float posy;
    float posz;
    float foax;
    float foay;
    float foaz;
    float roll;
  };
    
  /**
   * Camera interface
   */
  interface Camera extends ImageProvider
  {
    idempotent CameraDescription getCameraDescription();
    int setCameraDescription(CameraDescription description);
    
    string startCameraStreaming();
    
    void stopCameraStreaming();
    
    void reset();
    
  };

}; /*module*/

#endif /*CAMERA_ICE*/
