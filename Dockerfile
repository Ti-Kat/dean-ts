FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="tim.katzke@tu-dortmund.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# add algorithm implementation
COPY algorithm.py /app/
COPY src /app/src
