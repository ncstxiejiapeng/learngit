TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
    LIBS += -lpthread
    LIBS += -pthread

SOURCES += main.cpp \
    LongImageProcess.cpp \
    AutExp.cpp \
    ArmorDector.cpp \
    VideoCapture.cpp \
    LongFindArmor.cpp \
    AngleSolver.cpp

LIBS += /usr/local/lib/libopencv_calib3d.so \
/usr/local/lib/libopencv_core.so \
/usr/local/lib/libopencv_features2d.so \
/usr/local/lib/libopencv_flann.so \
/usr/local/lib/libopencv_highgui.so \
/usr/local/lib/libopencv_imgcodecs.so \
/usr/local/lib/libopencv_imgproc.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_objdetect.so \
/usr/local/lib/libopencv_photo.so \
/usr/local/lib/libopencv_shape.so \
/usr/local/lib/libopencv_stitching.so \
/usr/local/lib/libopencv_superres.so \
/usr/local/lib/libopencv_videoio.so \
/usr/local/lib/libopencv_video.so \
/usr/local/lib/libopencv_videostab.so

HEADERS += \
    LongImageProcess.h \
    AutExp.h \
    ArmorDector.h \
    VideoCapture.h \
    LongFindArmor.h \
    AngleSolver.h
