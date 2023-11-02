#include "utils.hpp"

std::vector<std::vector<int>> gridSquare(int x, int y, int w, int h, int num) {
    std::vector<std::vector<int>> coord;
    for (int i = 0; i <= num*2; i++) {
        for (int j = 0; j <= num*2; j++) {
            int xt = x-num+i;
            int yt = y-num+j;
            if (xt == x && yt == y) {

            }
            else {
                if (-num+i >= 0 && -num+j >= 0) {
                    if (!(xt >= w) && !(yt >= h)) {
                        coord.push_back({xt, yt});
                    }
                }
                else if (-num+i <= 0 && -num+j >= 0) {
                    if (!(xt <= -1) && !(yt >= h)) {
                        coord.push_back({xt, yt});
                    }
                }
                else if (-num+i >= 0 && -num+j <= 0) {
                    if (!(xt >= w) && !(yt <= -1)) {
                        coord.push_back({xt, yt});
                    }
                }
                else if (-num+i <= 0 && -num+j <= 0) {
                    if (!(xt <= -1) && !(yt <= -1)) {
                        coord.push_back({xt, yt});
                    }
                }
            }
        }
    }
/*
    for (int i = 0; i < num; i++) {
        if (!(x-1 <= -1)) {
            coord.push_back({x-1, y});
        }
        if (!(x+1 > w)) {
            coord.push_back({x+1, y});
        }
        if (!(y-1 <= -1)) {
            coord.push_back({x, y-1});
        }
        if (!(y+1 > h)) {
            coord.push_back({x, y+1});
        }

        if (!(x-1 <= -1) && !(y-1 <= -1)) {
            coord.push_back({x-1, y-1});
        }
        if (!(x-1 <=-1) && !(y+1 > h)) {
            coord.push_back({x-1, y+1});
        }
        if (!(x+1 > w) && !(y-1 <= -1)) {
            coord.push_back({x+1, y-1});
        }
        if (!(x+1 > w) && !(y+1 > h)) {
            coord.push_back({x+1, y-1});
        }
    }*/
    return coord;
}