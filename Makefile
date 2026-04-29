# Local dev and release: pytest (local or Docker), plugin zip for plugins.qgis.org
# Requires: git. For `package`, pass VERSION (tag or commit), e.g. make package VERSION=v1.0.0
# Excluded paths for the zip: add `path export-ignore` in .gitattributes (see git help gitattributes)

PLUGINNAME := ICEYE_toolbox
IMAGE      := qgis-test:latest
WORKDIR    := /plugins/$(PLUGINNAME)

USER_QGIS_PLUGINS := $(HOME)/.local/share/QGIS/QGIS3/profiles/default/python/plugins

ifeq ($(notdir $(CURDIR)),ICEYE_toolbox)
PLUGIN_PARENT ?= $(abspath $(CURDIR)/..)
else ifeq ($(shell test -e "$(USER_QGIS_PLUGINS)/ICEYE_toolbox" && echo yes),yes)
PLUGIN_PARENT ?= $(USER_QGIS_PLUGINS)
else
PLUGIN_PARENT ?= $(abspath $(CURDIR)/..)
endif

# Typical Debian/Ubuntu system QGIS install (override if needed)
QGIS_LOCAL_PY := /usr/share/qgis/python:/usr/share/qgis/python/plugins:/usr/lib/python3/dist-packages/qgis:/usr/share/qgis/python/qgis

QGIS_DOCKER_PY := /plugins:/usr/share/qgis/python/:/usr/share/qgis/python/plugins:/usr/lib/python3/dist-packages/qgis:/usr/share/qgis/python/qgis

# Optional override for install (linux, macos, darwin, windows, win); default from uname
INSTALL_OS ?= $(shell uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]')

.DEFAULT_GOAL := help

.PHONY: help docker-build image ensure-image test docker-test package install uninstall

help:
	@echo "Targets:"
	@echo "  make docker-build   Build $(IMAGE) from Dockerfile (run when Dockerfile or deps change)"
	@echo "  make test           Run pytest locally (PLUGIN_PARENT auto-set; needs QGIS Python)"
	@echo "  make docker-test    Run pytest in the image; builds $(IMAGE) only if missing"
	@echo "  make package VERSION=<tag|commit>   $(PLUGINNAME).zip via git archive (export-ignore in .gitattributes strips dev-only files)"
	@echo "  make install [INSTALL_OS=<os>]   Symlink repo into user QGIS plugins dir"
	@echo "  make uninstall [INSTALL_OS=<os>]   Remove that symlink (INSTALL_OS must match the install)"

# Alias
image: docker-build

docker-build:
	docker build -t $(IMAGE) -f Dockerfile .

ensure-image:
	@docker image inspect $(IMAGE) >/dev/null 2>&1 || { echo "Image $(IMAGE) not found; building..."; $(MAKE) docker-build; }

test:
	mkdir -p test-results
	QT_QPA_PLATFORM=offscreen \
	PYTHONPATH=$(PLUGIN_PARENT):$(QGIS_LOCAL_PY):$$PYTHONPATH \
	pytest test -v -s

docker-test: ensure-image
	mkdir -p test-results
	docker run --rm \
		--shm-size=2g \
		-v "$(CURDIR):$(WORKDIR)" \
		-e QT_QPA_PLATFORM=offscreen \
		-e PYTHONPATH=$(QGIS_DOCKER_PY) \
		-w $(WORKDIR) \
		$(IMAGE) \
		pytest test -v -s

package:
	@test -n "$(VERSION)" || (echo "Set VERSION, e.g. make package VERSION=v1.0.0" && false)
	rm -f $(PLUGINNAME).zip
	git archive --format=zip --prefix=$(PLUGINNAME)/ -o $(PLUGINNAME).zip $(VERSION)
	@echo "Created $(PLUGINNAME).zip from $(VERSION)"

install:
	@case "$(INSTALL_OS)" in \
	  linux) \
	    dst="$(HOME)/.local/share/QGIS/QGIS3/profiles/default/python/plugins" ;; \
	  macos|darwin) \
	    dst="$(HOME)/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins" ;; \
	  windows|win) \
	    dst="$$APPDATA/QGIS/QGIS3/profiles/default/python/plugins" ;; \
	  *) \
	    echo "Unknown OS: '$(INSTALL_OS)'. Use INSTALL_OS=linux, macos, or windows."; \
	    exit 1 ;; \
	esac; \
	echo "Target plugins directory: $$dst"; \
	if [ ! -d "$$dst" ]; then \
	  echo "Plugins directory doesn't exist. Creating it now..."; \
	  mkdir -p "$$dst"; \
	fi; \
	ln -sfn "$(CURDIR)" "$$dst/$(PLUGINNAME)"; \
	echo "All plugins linked for '$(INSTALL_OS)'."

# Remove symlink created by install
uninstall:
	@case "$(INSTALL_OS)" in \
	  linux) \
	    dst="$(HOME)/.local/share/QGIS/QGIS3/profiles/default/python/plugins" ;; \
	  macos|darwin) \
	    dst="$(HOME)/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins" ;; \
	  windows|win) \
	    dst="$$APPDATA/QGIS/QGIS3/profiles/default/python/plugins" ;; \
	  *) \
	    echo "Unknown OS: '$(INSTALL_OS)'. Use INSTALL_OS=linux, macos, or windows."; \
	    exit 1 ;; \
	esac; \
	link="$$dst/$(PLUGINNAME)"; \
	if [ -L "$$link" ]; then \
	  rm -f "$$link"; \
	  echo "Removed symlink $$link"; \
	elif [ -e "$$link" ]; then \
	  echo "$$link exists but is not a symlink; leaving it unchanged."; \
	else \
	  echo "Nothing to remove at $$link."; \
	fi
