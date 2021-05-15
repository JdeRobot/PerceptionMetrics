/*
 *  Copyright (C) 1997-2017 JDE Developers Team
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
 *  Authors :
 *       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
 */

#ifndef JDEROBOTCOMM_COMMUNICATOR_H
#define JDEROBOTCOMM_COMMUNICATOR_H

#ifdef ICE
#include <Ice/Communicator.h>
#include <Ice/Initialize.h>
#endif
#include <config/properties.hpp>
#include <comm/tools.hpp>


namespace Comm {

	class Communicator {
	public:
		Communicator(Config::Properties config);
		~Communicator();

		Config::Properties getConfig();
#ifdef ICE
		Ice::CommunicatorPtr getIceComm();
#endif

	private:
		Config::Properties config;
#ifdef ICE
		Ice::CommunicatorPtr ic;
#endif
	};


} //NS
#endif // JDEROBOTCOMM_COMMUNICATOR_H
