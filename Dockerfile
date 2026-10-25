FROM rust:1.94-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/chalk-statefulrouter /usr/local/bin/chalk-statefulrouter
EXPOSE 8080 8081
ENTRYPOINT ["chalk-statefulrouter"]
CMD ["--config", "/etc/chalk-statefulrouter/config.yaml"]
