
class FileHandler {
public:
	// TODO: ADD ERROR CHECKS TO ALL CALLERS FUNCTIONS THAT CALL THE FUNCTIONS IN THIS CLASS.
	// TODO: DON'T FORGET TO DELETE THE POINTER WHEN YOU FINISH WITH THE CLASS (IN THE CALLER FUNCTION).

	// Returns true if directory was created or already existed. Returns false if error occured.
	virtual bool CreateDir(const char* dir_name) = 0;
	// Returns the concatenation of directory name and filename. output should be in MAX_PATH length.
	virtual bool GetPath(const char* dir_name, const char* file_name, char* output) = 0;
	static FileHandler* Init();
};


class FileHandlerWindows : public FileHandler {
	virtual bool CreateDir(const char* dir_name);
	virtual bool GetPath(const char* dir_name, const char* file_name, char* output);
};

class FileHandlerLinux : public FileHandler {
	virtual bool CreateDir(const char* dir_name);
	virtual bool GetPath(const char* dir_name, const char* file_name, char* output);
};
