#!/bin/bash
#
# Copyright 2020 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Purpose: This script installs NVIDIA Drivers for GPU
#
# Refer the following links for NVIDIA driver installation.
# https://developer.nvidia.com/cuda-toolkit-archive
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/"
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

export ENV_FILE="/etc/profile.d/env.sh"
# shellcheck source=/etc/profile.d/env.sh disable=SC1091
source "${ENV_FILE}" || exit 1


function get_metadata_value() {
  curl --retry 5 \
    -s \
    -f \
    -H "Metadata-Flavor: Google" \
    "http://metadata/computeMetadata/v1/$1"
}

function get_attribute_value() {
  get_metadata_value "instance/attributes/$1"
}

function install_linux_headers() {
  # Install linux headers. Note that the kernel version might be changed after
  # installing gvnic version. For example: 4.19.0-8-cloud-amd64 ->
  # 4.19.0-9-cloud-amd64. So we install the kernel headers for each driver
  # installation.
  echo "install linux headers: linux-headers-$(uname -r)"
  sudo apt install -y linux-headers-"$(uname -r)" || exit 1
}

# Try to download driver via Web if GCS failed (Example: VPC-SC/GCS failure)
function download_driver_via_http() {
  local driver_url_path=$1
  local downloaded_file=$2
  echo "Could not use Google Cloud Storage APIs to download driver. Attempting to download them directly from Nvidia."
  echo "Downloading driver from URL: ${driver_url_path}"
  wget -nv "${driver_url_path}" -O "${downloaded_file}" || {
    echo 'Download driver via Web failed!' &&
    rm -f "${downloaded_file}" &&
    echo "${downloaded_file} deleted"
  }
}

# For Debian-like OS
function install_driver_debian() {
  echo "DRIVER_VERSION: ${DRIVER_VERSION}"
  local driver_installer_file_name="driver_installer.run"
  local nvidia_driver_file_name="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
  if [[ -z "${DRIVER_GCS_PATH}" ]]; then
    DRIVER_GCS_PATH="gs://nvidia-drivers-us-public/tesla/${DRIVER_VERSION}"
  fi
  local driver_gcs_file_path=${DRIVER_GCS_PATH}/${nvidia_driver_file_name}
  echo "Downloading driver from GCS location and install: ${driver_gcs_file_path}"
  set +e
  gsutil -q cp "${driver_gcs_file_path}" "${driver_installer_file_name}"
  set -e
  # Download driver via http if GCS failed.
  if [[ ! -f "${driver_installer_file_name}" ]]; then
    driver_url_path="http://us.download.nvidia.com/tesla/${DRIVER_VERSION}/${nvidia_driver_file_name}"
    download_driver_via_http "${driver_url_path}" "${driver_installer_file_name}"
  fi

  if [[ ! -f "${driver_installer_file_name}" ]]; then
    echo "Failed to find drivers!"
    exit 1
  fi

  chmod +x ${driver_installer_file_name}
  sudo ./${driver_installer_file_name} --dkms -a -s --no-drm --install-libglvnd
  rm -rf ${driver_installer_file_name}
}

# For Ubuntu OS
function install_driver_ubuntu() {
  echo "DRIVER_UBUNTU_DEB: ${DRIVER_UBUNTU_DEB}"
  echo "DRIVER_UBUNTU_PKG: ${DRIVER_UBUNTU_PKG}"
  if [[ -z "${DRIVER_GCS_PATH}" ]]; then
    DRIVER_GCS_PATH="gs://dl-platform-public-nvidia/${DRIVER_UBUNTU_DEB}"
  fi
  echo "Downloading driver from GCS location and install: ${DRIVER_GCS_PATH}"
  set +e
  gsutil -q cp "${DRIVER_GCS_PATH}" "${DRIVER_UBUNTU_DEB}"
  set -e
  # Download driver via http if GCS failed.
  if [[ ! -f "${DRIVER_UBUNTU_DEB}" ]]; then
    driver_url_path="https://developer.download.nvidia.com/compute/cuda/${DRIVER_UBUNTU_CUDA_VERSION}/local_installers/${DRIVER_UBUNTU_DEB}"
    download_driver_via_http "${driver_url_path}" "${DRIVER_UBUNTU_DEB}"
  fi
  if [[ ! -f "${DRIVER_UBUNTU_DEB}" ]]; then
    driver_url_path="https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/${DRIVER_UBUNTU_DEB}"
    download_driver_via_http "${driver_url_path}" "${DRIVER_UBUNTU_DEB}"
  fi
  if [[ ! -f "${DRIVER_UBUNTU_DEB}" ]]; then
    echo "Failed to find drivers!"
    exit 1
  fi
  wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

  sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
  dpkg -i "${DRIVER_UBUNTU_DEB}" || {
    echo "Failed to install ${DRIVER_UBUNTU_DEB}..exit"
    exit 1
  }
  apt-key add /var/cuda-repo-*/*.pub || apt-key add /var/nvidia-driver*/*.pub || {
    echo "Failed to add apt-key...exit"
    exit 1
  }
  sudo apt-get --allow-releaseinfo-change update
  sudo apt remove -y "${DRIVER_UBUNTU_PKG}"
  sudo apt -y autoremove && sudo apt install -y "${DRIVER_UBUNTU_PKG}" nvidia-modprobe
  rm -rf "${DRIVER_UBUNTU_DEB}" cuda-update1804.pin
}

function wait_apt_locks_released() {
  # Wait for apt lock to be released
  # Source: https://askubuntu.com/a/373478
  echo "wait apt locks released"
  while sudo fuser /var/{lib/{dpkg,apt/lists},cache/apt/archives}/lock >/dev/null 2>&1 ||
     sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
     sleep 1
  done
}

main() {
  wait_apt_locks_released
  install_linux_headers
  # shellcheck source=/opt/deeplearning/driver-version.sh disable=SC1091
  source "${DL_PATH}/driver-version.sh"
  export DRIVER_GCS_PATH
  # Custom GCS driver location via instance metadata.
  DRIVER_GCS_PATH=$(get_attribute_value nvidia-driver-gcs-path)
  if [[ "${OS_IMAGE_FAMILY}" == "${OS_DEBIAN9}" || "${OS_IMAGE_FAMILY}" == "${OS_DEBIAN10}" ]]; then
    install_driver_debian
  elif [[ "${OS_IMAGE_FAMILY}" == "${OS_UBUNTU1804}" || "${OS_IMAGE_FAMILY}" == "${OS_UBUNTU2004}" ]]; then
    install_driver_ubuntu
  fi
  exit 0
}

main
