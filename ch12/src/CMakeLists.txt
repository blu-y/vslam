add_library(mySLAM SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
#        config.cpp
        feature.cpp
#        frontend.cpp
#        viewer.cpp
#        visual_odometry.cpp
#        dataset.cpp
)

target_link_libraries(mySLAM ${THIRD_PARTY_LIBS})