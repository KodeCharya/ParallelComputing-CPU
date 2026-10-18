#include "parallel_compute/system/system_info.hpp"
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    bool json_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--json") {
            json_mode = true;
        }
    }

    const auto& sys = parallel_compute::SystemInfo::get();
    if (json_mode) {
        std::cout << sys.to_json() << "\n";
    } else {
        std::cout << "=== ParallelComputeLab System Diagnostics ===\n"
                  << sys.to_string() << "\n";
    }
    return 0;
}
