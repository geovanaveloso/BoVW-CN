#-------------------------------------------------
#
# Project created by QtCreator 2015-11-05T18:52:21
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = BOC
CONFIG   += console
CONFIG   -= app_bundle
QMAKE_CFLAGS_RELEASE += -o -fopenmp
QMAKE_CFLAGS_DEBUG += -o -fopenmp
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS +=  -fopenmp

TEMPLATE = app

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_legacy -lopencv_nonfree -lopencv_ml -lopencv_flann

INCLUDEPATH += $$PWD/../../../../../../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../../../../../../usr/local/include

SOURCES += \
    BOWProperties.cpp \
    Utils.cpp \
    Image.cpp \
    Group.cpp \
    Classifier.cpp \
    Main.cpp \
    Dataset.cpp \
    Descriptors.cpp \
    Graph.cpp

HEADERS += \
    BOWProperties.h \
    Utils.h \
    Image.h \
    Group.h \
    Classifier.h \
    Dataset.h \
    Descriptors.h \
    Graph.h


win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../../../../usr/local/include/LibOPF/lib/release/ -lOPF
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../../../usr/local/include/LibOPF/lib/debug/ -lOPF
else:unix: LIBS += -L$$PWD/../../../../../../../../usr/local/include/LibOPF/lib/ -lOPF

INCLUDEPATH += $$PWD/../../../../../../../../usr/local/include/LibOPF/include
DEPENDPATH += $$PWD/../../../../../../../../usr/local/include/LibOPF/include

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/include/LibOPF/lib/release/libOPF.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/include/LibOPF/lib/debug/libOPF.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/include/LibOPF/lib/release/OPF.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/include/LibOPF/lib/debug/OPF.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/include/LibOPF/lib/libOPF.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../../../../usr/local/lib/release/ -ligraph
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../../../usr/local/lib/debug/ -ligraph
else:unix: LIBS += -L$$PWD/../../../../../../../../usr/local/lib/ -ligraph

INCLUDEPATH += $$PWD/../../../../../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../../../../../usr/local/include

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/release/libigraph.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/debug/libigraph.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/release/igraph.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/debug/igraph.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/libigraph.a


win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../../../../usr/local/lib/release/ -lopencv_ts
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../../../usr/local/lib/debug/ -lopencv_ts
else:unix: LIBS += -L$$PWD/../../../../../../../../usr/local/lib/ -lopencv_ts

INCLUDEPATH += $$PWD/../../../../../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../../../../../usr/local/include

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/release/libopencv_ts.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/debug/libopencv_ts.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/release/opencv_ts.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/debug/opencv_ts.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../../../../usr/local/lib/libopencv_ts.a
