# SPDX-FileCopyrightText: 2026 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG RELEASE
ARG LAUNCHPAD_BUILD_ARCH
ENV USER=user1
ARG PASSWORD=user1
SHELL ["/bin/bash", "-c"]
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA="1"

RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime && \
    apt update &&\
    apt install --no-install-recommends -y -qq sudo git cmake && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup ${USER} \
    && useradd -ms /bin/bash ${USER} -g ${USER} \
    && echo "${USER}:${PASSWORD}" | chpasswd \
    && usermod -a -G sudo ${USER} \
    && sed -i.bak -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

RUN sudo apt update && DEBIAN_FRONTEND=noninteractive sudo apt install ffmpeg libsm6 libxext6 -y

# ROS2 Install
ENV ROS_DISTRO=jazzy
RUN sudo apt update && sudo apt install -y locales && sudo locale-gen en_US en_US.UTF-8 \
    && sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8

RUN sudo apt install software-properties-common -y\
    && sudo add-apt-repository universe \
    && sudo apt update && sudo apt install curl -y

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    sudo apt install /tmp/ros2-apt-source.deb

RUN sudo apt update && sudo apt install ros-$ROS_DISTRO-ros-base \
    ros-$ROS_DISTRO-sensor-msgs ros-$ROS_DISTRO-geometry-msgs \
    ros-$ROS_DISTRO-action-msgs ros-$ROS_DISTRO-actionlib-msgs ros-$ROS_DISTRO-nav2-rviz-plugins \
    ros-$ROS_DISTRO-nav2-msgs ros-$ROS_DISTRO-nav2-simple-commander ros-$ROS_DISTRO-rviz2 ros-dev-tools ros-$ROS_DISTRO-test-msgs -y

RUN sudo apt update && sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp -y
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USER}/.bashrc

# VS code
RUN wget -O- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor | sudo tee /usr/share/keyrings/vscode.gpg && \
    echo deb [arch=amd64 signed-by=/usr/share/keyrings/vscode.gpg] https://packages.microsoft.com/repos/vscode stable main | sudo tee /etc/apt/sources.list.d/vscode.list
RUN sudo apt update && sudo apt install -y code
# Install VisualStudio Code extensions
RUN code --install-extension ms-vscode.cpptools \
		 --install-extension ms-vscode.cpptools-themes \
		 --install-extension ms-vscode.cmake-tools \
         --install-extension ms-python.python \
         --install-extension eamodio.gitlens

# Git Setup
ARG GIT_USERNAME
ARG GIT_USER_EMAIL
ARG GIT_TOKEN
RUN git config --global user.name ${GIT_USERNAME} && git config --global user.email ${GIT_USER_EMAIL}
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USER}/.bashrc

# YARP Install
RUN sudo apt-get install -y build-essential cmake-curses-gui \
    libeigen3-dev \
    libace-dev \
    libedit-dev \
    libsqlite3-dev \
    libtinyxml-dev \
    qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
    qml-module-qtquick2 qml-module-qtquick-window2 \
    qml-module-qtmultimedia qml-module-qtquick-dialogs \
    qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel \
    qml-module-qt-labs-settings \
    libqcustomplot-dev \
    libgraphviz-dev \
    libjpeg-dev \
    gedit \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    swig
RUN git clone https://github.com/robotology/ycm-cmake-modules && cd ycm-cmake-modules && mkdir build && cd build && cmake .. && sudo make install -j8
COPY ./fix_swig_yarp.sh /home/${USER}/
RUN sudo apt install -y libace-dev gdb unzip
RUN bash /home/${USER}/fix_swig_yarp.sh  &&\
    git clone https://github.com/robotology/yarp && \
    cd yarp && git switch yarp-3.12 && mkdir build &&\
    cd build && \
    cmake -DYARP_COMPILE_BINDINGS=ON \
    -DCREATE_PYTHON=ON \
    -DSWIG_EXECUTABLE=/usr/bin/swig \
    -DSWIG_DIR=/usr/share/swig/4.3.0 \
    -DENABLE_yarpcar_mjpeg=ON \
    .. && sudo make install -j12
ENV YARP_COLORED_OUTPUT=1
ENV YARP_DATA_DIRS=/usr/local/share/yarp
RUN sudo ln -s /usr/local/share/bash-completion/completions/yarp /usr/share/bash-completion/completions && \
    sudo apt install -y glpk-doc glpk-utils libglpk-dev libglpk40

# Yarp devices ros2
RUN git clone https://github.com/robotology/yarp-devices-ros2 && \
    cd yarp-devices-ros2/ros2_interfaces_ws && \
    git checkout v2.0.0 && \
    source /opt/ros/$ROS_DISTRO/setup.sh && colcon build && \
    cd .. && mkdir build && cd build && \
    source /opt/ros/$ROS_DISTRO/setup.bash && source /home/$USER/yarp-devices-ros2/ros2_interfaces_ws/install/setup.bash &&\
    cmake -DYARP_ROS2_USE_SYSTEM_map2d_nws_ros2_msgs=ON \
    -DYARP_ROS2_USE_SYSTEM_yarp_control_msgs=ON \
    .. && make -j8 && \
    echo "source /home/$USER/yarp-devices-ros2/ros2_interfaces_ws/install/local_setup.bash" >> ~/.bashrc
ENV YARP_DATA_DIRS=${YARP_DATA_DIRS}:/home/$USER/yarp-devices-ros2/build/share/yarp:/home/$USER/yarp-devices-ros2/build/share/yarp-devices-ros2

### Lerobot install
RUN sudo apt update && sudo apt install python3-venv python3-pip -y
RUN git clone https://${GIT_TOKEN}@github.com/hsp-iit/lerobot && \
    cd lerobot && pip install -e '.[dataset]' --break-system-packages

### MoGe
RUN git clone https://github.com/SimoneMic/MoGe && \
    cd MoGe && pip install -r requirements.txt --break-system-packages &&\
    pip install -e . --break-system-packages
ENV PATH=$PATH:/home/$USER/.local/bin

# Cleanup
RUN sudo apt update && sudo apt install -y iproute2 unzip terminator bash-completion locate
