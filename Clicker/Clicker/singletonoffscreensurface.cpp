//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//  
//  This file is part of the Artiatomi package.
//  
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//  
////////////////////////////////////////////////////////////////////////


#include "singletonoffscreensurface.h"

QOffscreenSurface* SingletonOffscreenSurface::Get()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    if (!_instance)
    {
        _instance = new SingletonOffscreenSurface();

        _instance->_surface = new QOffscreenSurface();
        _instance->_surface->create();
    }
    return _instance->_surface;
}

void SingletonOffscreenSurface::Cleanup()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    if (_instance)
    {
        _instance->_surface->destroy();
        delete _instance->_surface;
        delete _instance;
        _instance = NULL;
    }
}

SingletonOffscreenSurface* SingletonOffscreenSurface::_instance = NULL;
std::recursive_mutex SingletonOffscreenSurface::_mutex;
