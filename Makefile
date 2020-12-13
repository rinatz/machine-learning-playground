PROJECT_DIR = $(CURDIR)
CONAN_INSTALL_DIR = $(PROJECT_DIR)/build
PROTOC = $(CONAN_INSTALL_DIR)/bin/protoc
TF_MODELS_DIR = $(PROJECT_DIR)/models

.PHONY: install
install:
	poetry install
	mkdir -p $(CONAN_INSTALL_DIR)
	cd $(CONAN_INSTALL_DIR) && poetry run conan install $(PROJECT_DIR)
	git clone https://github.com/tensorflow/models
	cd $(TF_MODELS_DIR)/research && \
		$(PROTOC) object_detection/protos/*.proto --python_out=. && \
		cp object_detection/packages/tf2/setup.py .
	poetry run python -m pip install --use-feature=2020-resolver $(TF_MODELS_DIR)/research
