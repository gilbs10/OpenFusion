#ifndef ___UTILITIES_H_
#define ___UTILITIES_H_

//#include <math.h>

// openCV Includes
//#include <highgui.h>

#include "Constants.h"
#include "BasicTypes.h"

static int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


#endif // ___UTILITIES_H_


// TODO - REMOVE THIS
/////////////////////////// CONVENTIONS //////////////////////////
// - Classes and Function: TitleCase
// - Methods: Use verbs
// - Member fields: m_nameNameName
// - Member device fields: md_nameNameName
// - Member methods: TitleCase
// - Arguments in Functions: nameNameName
// - Device Arguments in Functions: d_nameNameName
// - CONST EVERYWHERE
// - Constants in uppercase
// - Doxygen Comments
// - 100 letter lines
// - Egyptian brackets