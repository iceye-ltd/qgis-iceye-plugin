ARG QGIS_VERSION=3.44
FROM qgis/qgis:${QGIS_VERSION}

RUN apt-get update && apt-get install -y python3-pip xvfb

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /