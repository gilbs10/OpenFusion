/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
/* CUda UTility Library */

/* Credit: Cuda team for the PGM file reader / writer code. */

// includes, file
#include <cutil.h>

// includes, system
#include <iostream>

// includes, common
#include "stopwatch.h"

#ifndef max
#define max(a,b) (a < b ? b : a);
#endif

#ifndef min
#define min(a,b) (a < b ? a : b);
#endif

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return CUTTrue if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutCreateTimer( unsigned int* name) 
{
    *name = StopWatch::create();
    return (0 != name) ? CUTTrue : CUTFalse;
}


////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return CUTTrue if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutDeleteTimer( unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

    try 
    {
        StopWatch::destroy( name);
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutStartTimer( const unsigned int name) 
{
    CUTBoolean retval = CUTTrue;
    StopWatch::get( name).start();
    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutStopTimer( const unsigned int name) 
{
    CUTBoolean retval = CUTTrue;
    StopWatch::get( name).stop();
    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutResetTimer( const unsigned int name)
{
    CUTBoolean retval = CUTTrue;
	StopWatch::get( name).reset();
    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer 
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
float CUTIL_API
cutGetAverageTimerValue( const unsigned int name)
{
    float time = 0.0;
    time = StopWatch::get( name).getAverageTime();
    return time;
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
float CUTIL_API
cutGetTimerValue( const unsigned int name) 
{  
    float time = 0.0;
    time = StopWatch::get( name).getTime();
    return time;
}
