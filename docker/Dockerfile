FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    libxml2-dev \
    libgsl-dev \
    libglu1-mesa \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxft2 \
    libxinerama1 \
    libxrandr2 \
    libxi6 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/Discngine/fpocket.git && \
    cd fpocket && \
    make && \
    make install && \
    cd .. && \
    rm -rf fpocket

WORKDIR /pockets
COPY requirements.txt .
    
RUN pip install -r requirements.txt

CMD ["echo" "ImageCreated"]
