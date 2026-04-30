# Local dev and release: pytest (local or Docker), plugin zip for plugins.qgis.org
# Requires: git. For `package`, pass VERSION (tag or commit), e.g. make package VERSION=v1.0.0
# Excluded paths for the zip: add `path export-ignore` in .gitattributes (see git help gitattributes)

PLUGINNAME := iceye_toolbox

QGIS_VERSION ?= 3.44
WORKSPACE ?= $(shell pwd)

.DEFAULT_GOAL := help

.PHONY: help docker-build test package install uninstall

help:
	@echo "Targets:"
	@echo "  make test           Run pytest locally (PLUGIN_PARENT auto-set; needs QGIS Python)"
	@echo "  make package VERSION=<tag|commit>   $(PLUGINNAME).zip via git archive (export-ignore in .gitattributes strips dev-only files)"
	@echo "  make install [INSTALL_OS=<os>]   Symlink repo into user QGIS plugins dir"
	@echo "  make uninstall [INSTALL_OS=<os>]   Remove that symlink (INSTALL_OS must match the install)"

test:
	QGIS_VERSION=$(QGIS_VERSION) WORKSPACE=$(WORKSPACE) docker-compose -f docker-compose.yml run qgis /usr/src/iceye_toolbox/test.sh

package:
	@test -n "$(VERSION)" || (echo "Set VERSION, e.g. make package VERSION=v1.0.0" && false)
	rm -f $(PLUGINNAME).zip
	git archive --format=zip --prefix=$(PLUGINNAME)/ -o $(PLUGINNAME).zip $(VERSION)
	@echo "Created $(PLUGINNAME).zip from $(VERSION)"

install:
	./install.sh

# Remove symlink created by install
uninstall:
	./uninstall.sh