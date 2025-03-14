#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#include <windows.h>
#else
#include <libgen.h>
#include <limits.h>
#include <unistd.h>
#endif

void set_pwd_to_exe_dir(void) {
  char exePath[PATH_MAX];

#ifdef _WIN32
  // Get the full path of the executable
  DWORD length = GetModuleFileNameA(NULL, exePath, sizeof(exePath));
  if (length == 0 || length == sizeof(exePath)) {
    fprintf(stderr, "Error getting executable path\n");
  }
  // Windows does not provide dirname, so we use _splitpath.
  char drive[_MAX_DRIVE], dir[_MAX_DIR];
  _splitpath(exePath, drive, dir, NULL, NULL);
  char exeDir[PATH_MAX];
  snprintf(exeDir, sizeof(exeDir), "%s%s", drive, dir);
  // Change the current working directory
  if (_chdir(exeDir) != 0) {
    perror("chdir");
  }
#else
  // On Linux, readlink the /proc/self/exe to get the executable path.
  ssize_t count = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
  if (count == -1) {
    perror("readlink");
  }
  exePath[count] = '\0';
  // Use dirname to extract the directory from the path.
  char *dir = dirname(exePath);
  if (chdir(dir) != 0) {
    perror("chdir");
  }
#endif
}
