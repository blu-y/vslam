add_library(mySLAM SHARED
        camera.cpp
        config.cpp
        dataset.cpp
        frame.cpp
        feature.cpp
        map.cpp
        mappoint.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
)

target_link_libraries(mySLAM ${THIRD_PARTY_LIBS})