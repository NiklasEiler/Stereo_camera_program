#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main() {
    char cwd[1024]; // Buffer to store the current directory path

    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        char python_str[] = "/.venv/Scripts/python.exe";
        char space[] = " ";
        char gui[] = "/gui.py";
        char command[1000];

        strcpy(command, cwd);
        strcat(command, python_str);
        strcat(command, space);
        strcat(command, cwd);
        strcat(command, gui);
        //printf(command);
        system(command);

    } else {
        perror("getcwd() error");
    }
    sleep(60);
}

