#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>

static uint32_t float_to_ordered_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if (u & 0x80000000u) {
        u = 0x80000000u - u;
    } else {
        u = u + 0x80000000u;
    }
    return u;
}

unsigned long ulps_between(float a, float b) {
    uint32_t ua = float_to_ordered_uint(a);
    uint32_t ub = float_to_ordered_uint(b);

    uint32_t diff = (ua > ub) ? (ua - ub) : (ub - ua);
    return static_cast<unsigned long>(diff);
}

void test_ulps(float start, float end) {
    unsigned long ulps_dist = ulps_between(start, end);
    std::cout << "Test dla a = " << start << ", b = " << end << std::endl;
    std::cout << "ulps_between() zwraca: " << ulps_dist << std::endl;

    unsigned long steps = 0;
    float current = start;

    if (start < end) {
        while (current < end && steps <= ulps_dist) {
            current = nextafterf(current, end);
            steps++;
        }
    } else if (start > end) {
        while (current > end && steps <= ulps_dist) {
            current = nextafterf(current, end);
            steps++;
        }
    }

    std::cout << "Liczba krokow z nextafterf(): " << steps << std::endl;
    if (ulps_dist == steps) {
        std::cout << "Wynik poprawny!" << std::endl;
    } else {
        std::cout << "Blad! Wyniki sie nie zgadzaja." << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}

int main() {
    test_ulps(1.0f, 1.000001f);
    test_ulps(10.0f, nextafterf(10.0f, 11.0f));
    // test_ulps(-0.0001f, 0.0002f);
    test_ulps(1024.f, 1024.001f);

    return 0;
}

