docker buildx build --push \
     --tag gouthamshiv/ml-classifier:latest \
     --tag gouthamshiv/ml-classifier:0.1 \
     --platform linux/arm64,linux/amd64 .
