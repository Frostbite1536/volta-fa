add_library(volta_fa_p2p STATIC p2p.cpp)
target_include_directories(volta_fa_p2p PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(volta_fa_p2p PUBLIC ${VFA_CUDART_TARGET})
