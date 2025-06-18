# Docker setup

This docker setup is tested on Windows 10.

make sure you are under this directory yourworkspace/PartPacker/docker

Build docker image:

```
docker build -t partpacker:latest .
```

Run docker image at the first time:

```
docker run --name partpacker --gpus all -it -p 7860:7860 partpacker python app.py
```

After first time:
```
docker start -a partpacker
```

Stop the container:
```
docker stop partpacker
```

You can find the demo link showing in terminal, such as `https://94fc1ba77a08526e17.gradio.live/` or something similar else (it will be changed after each time to restart the container) to use the demo.