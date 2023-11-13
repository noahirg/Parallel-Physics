//#include <vector>
#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP
#include <cmath>
#include <iostream>
#include <vector>

const double pi = 3.14159265358979323846;

std::vector<std::vector<int>> gridSquare(int x, int y, int w, int h, int num);

//namespace ut

//{
template <typename T>
class Vec2
{
    public:
    T x;
    T y;
    Vec2() : x(T{}), y(T{}) {}

    Vec2(T X, T Y) : x(X), y(Y) {}

    template <typename U>
    Vec2(const Vec2<U> &vector) : x(static_cast<T>(vector.x)), y(static_cast<T>(vector.y)) {}

    void setMagn(T mag) {
        Vec2<T> tem = Vec2<T>(this->x, this->y) / (T)hypotf(this->x, this->y);
        this->x = tem.x * mag;
        this->y = tem.y * mag;
    }

    T magnSq() {
        return (this->x * this->x) + (this->y * this->y);
    }

    T magn() {
        return sqrt(magnSq());
    }

    void normalize() {
        Vec2<T> tem = Vec2<T>(this->x, this->y) / (T)hypotf(this->x, this->y);
        this->x = tem.x;
        this->y = tem.y;
    }

    T heading(std::string type = "") {
        float d = atan2(this->y, this->x);
        if (d < 0) {d += 2 * pi;}
        if (type == "degree") {
            d = atan2(this->y, this->x) * 180 / pi;
            if (d < 0) {d += 360;}
        }
        return d;
    }

    void setHeading(float ang, std::string type = "") {
        float d = ang;
        if (type == "degree") {d = ang * pi / 180;}
        float m = magn();
        this->x = m * cos(d);
        this->y = m * sin(d);
    }

    float dot(Vec2<T> &vec) {
        return this->x * vec.x + this->y * vec.y;
    }

    void limit(float mag) {
        if (mag * mag < magnSq()) {
            Vec2<T> tem = Vec2<T>(this->x, this->y) / (T)hypotf(this->x, this->y);
            this->x = tem.x * mag;
            this->y = tem.y * mag;
        }
    }

    void rotate(float ang, std::string type = "") {
        float newHead = heading() + ang;
        if (type == "degree") {
            newHead = heading(type) + ang;
        }
        setHeading(newHead, type);
    }

    float angleBetween(Vec2<T> &vec, std::string type = "") {
        float av = vec.heading();
        float at = 0.f;
        if (av > pi) {
            at = 2 * pi;
        }
        float ang = av - at;
        if (type == "degree") {ang = ang * 180 / pi;}
        return ang;
    }

    void lerp(T x, T y, float amt) {
        this->x += (x - this->x) * amt;
        this->y += (y - this->y) * amt;
    }

    Vec2<T> copyOf() {
        return Vec2<T>(x, y);
    }

    static Vec2<T> normalize(Vec2<T> &v1) {
        return Vec2<T>(v1.x, v1.y) / (T)hypotf(v1.x, v1.y); 
    }

    static Vec2<T> lerp(Vec2<T> &v1, Vec2<T> &v2, float amt) {
        //use by doing Vec2f::lerp
        float x = v1.x + ((v2.x - v1.x) * amt);
        float y = v1.y + ((v2.x - v1.x) * amt);
        return Vec2<T>(x, y);
    }

    static Vec2<T> max(Vec2<T> &v1, Vec2<T> &v2)
    {
        //use by doing Vec2<T>::max
        if (v1 > v2)
            return v1;
        return v2;
    }

    static Vec2<T> min(Vec2<T> &v1, Vec2<T> &v2)
    {
        //use by doing Vec2<T>::max
        if (v1 < v2)
            return v2;
        return v1;
    }
};

template <typename T>
Vec2<T> operator -(const Vec2<T> &right) {
    return Vec2<T>(-right.x, -right.y);
}

template <typename T>
Vec2<T> &operator +=(Vec2<T> &left, const Vec2<T> &right) {
    left.x += right.x;
    left.y += right.y;
    return left;
}

template <typename T>
Vec2<T> &operator -=(Vec2<T> &left, const Vec2<T> &right) {
    left.x -= right.x;
    left.y -= right.y;
    return left;
}

template <typename T>
Vec2<T> operator +(const Vec2<T> &left, const Vec2<T> &right) {
    return Vec2<T>(left.x+right.x, left.y+right.y);
}

template <typename T>
Vec2<T> operator -(const Vec2<T> &left, const Vec2<T> &right) {
    return Vec2<T>(left.x-right.x, left.y-right.y);
}

template <typename T>
Vec2<T> operator *(const Vec2<T> &left, T right) {
    return Vec2<T>(left.x * right, left.y * right);
}

template <typename T>
Vec2<T> operator *(T left, const Vec2<T> &right) {
    return Vec2<T>(right.x * left, right.y * left);
}

template <typename T>
Vec2<T> &operator *=(Vec2<T> &left, T right) {
    left.x *= right;
    left.y *= right;
    return left;
}

template <typename T>
Vec2<T> operator /(const Vec2<T> &left, T right) {
    return Vec2<T>(left.x / right, left.y / right);
}

template <typename T>
Vec2<T> operator /=(Vec2<T> &left,T right) {
    left.x /= right;
    left.y /= left;
    return left;
}

template <typename T>
bool operator ==(const Vec2<T> &left, const Vec2<T> &right) {
    return (left.x == right.x) && (left.y == right.y);
}

template <typename T>
bool operator !=(const Vec2<T> &left, const Vec2<T> &right) {
    return (left.x != right.x) || (left.y != right.y);
}

template <typename T>
bool operator >(const Vec2<T> &left, const Vec2<T> &right) {
    return (left.x * left.x + left.y * left.y) > (right.x * right.x + right.y * right.y);
}

template <typename T>
bool operator <(const Vec2<T> &left, const Vec2<T> &right) {
    return (left.x * left.x + left.y * left.y) < (right.x * right.x + right.y * right.y);
}

template <typename T>
bool operator >=(const Vec2<T> &left, const Vec2<T> &right) {
    return (left.x * left.x + left.y * left.y) >= (right.x * right.x + right.y * right.y);
}

template <typename T>
bool operator <=(const Vec2<T> &left, const Vec2<T> &right) {
    return (left.x * left.x + left.y * left.y) <= (right.x * right.x + right.y * right.y);
}

typedef Vec2<int>     Vec2i;
typedef Vec2<float>   Vec2f;



//check and return square around specific spot in grid / 2d vector
//x and y are center location
//w and h is lengths of grid
//num is how many squares checked
std::vector<std::vector<int>> gridSquare(int x, int y, int w, int h, int num); /*{
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
    }*8/
    return coord;
}*/

#endif


/*
void setMagn(sf::Vector2f &v, float mag) {
    v = v / hypotf(v.x, v.y);
    v = v * mag;
}*/