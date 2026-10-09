add_executable(p2p_bandwidth p2p_bandwidth.cpp)
find_package(Threads REQUIRED)
target_link_libraries(p2p_bandwidth PRIVATE volta_fa_p2p Threads::Threads)
