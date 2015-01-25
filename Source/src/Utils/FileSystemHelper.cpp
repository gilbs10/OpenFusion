#include "FileSystemHelper.h"
#include <string.h>
#include <cstdio>


#if defined(_WIN32) || defined(_WIN64)
	// Windows
	#include <windows.h> // for CreateDirectory
	#include "Shlwapi.h" // for PathAppend
	
	FileHandler* FileHandler::Init() {
		return new FileHandlerWindows();
	}

	bool FileHandlerWindows::CreateDir(const char* dir_name) {
		int status = CreateDirectory(dir_name, NULL);
		if (!status) { // Calling WinBase.h
			return (GetLastError() == ERROR_ALREADY_EXISTS); // Returns ok if path exists
		}
		return true;
	}

	bool FileHandlerWindows::GetPath(const char* dir_name, const char* file_name, char* output) {
		strcpy(output, dir_name);
		int status = PathAppend(output, file_name);
		return status;
	}
	
#else
	// Linux
	#include <sys/stat.h>
	#include <sys/types.h>
	#include <errno.h>

	FileHandler* FileHandler::Init() {
		return new FileHandlerLinux();
	}

	bool FileHandlerLinux::CreateDir(const char* dir_name) {
		int status = mkdir(dir_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (!status) {
			return (errno == EEXIST); // Returns ok if path exists
		}
		return true;
	}

	bool FileHandlerLinux::GetPath(const char* dir_name, const char* file_name, char* output) {
		sprintf(output, "%s%c%s", dir_name, '/', file_name);
		return true;
	}


#endif
