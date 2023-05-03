FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="tim.katzke@tu-dortmund.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY timeeval/requirements.txt /app/
RUN pip install -r /app/requirements.txt

# add algorithm implementation
COPY timeeval/algorithm.py /app/
COPY src /app/src
