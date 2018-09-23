#pragma once

#include <cstddef>
#include <cstdint>

namespace memb {

struct PrefixCode {
    uint16_t code;
    size_t bitsCount;
};

}
