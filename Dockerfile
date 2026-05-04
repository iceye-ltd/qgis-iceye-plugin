ARG QGIS_VERSION=3.44
FROM qgis/qgis:${QGIS_VERSION}

# Core Python bindings live under /usr/share/qgis/python; the built-in Processing
# plugin (module name `processing`) lives under python/plugins and is not on the
# default path — pytest imports processing.core.Processing before full GUI startup.
ENV PYTHONPATH=/usr/share/qgis/python:/usr/share/qgis/python/plugins

RUN apt-get update && apt-get install -y python3-pip xvfb && apt remove -y python3-pytest

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

WORKDIR /