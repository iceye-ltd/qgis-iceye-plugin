FROM qgis/qgis:release-3_34

USER root

ENV PYTHONPATH="/plugins:${PYTHONPATH}"
ENV DISPLAY=:99

RUN apt-get update && apt-get install -y python3-pip xvfb

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY . /plugins/ICEYE_toolbox

WORKDIR /plugins/ICEYE_toolbox