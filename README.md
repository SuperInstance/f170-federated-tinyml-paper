# F170 — Federated TinyML for the Vessel Edge

The on-device learning loop for the vessel-agent system.

Every device on a fishing boat runs the same frozen audio backbone
and learns a tiny classifier head on its own audio. Once a round,
the device ships its head to the aggregator. The aggregator
FedAvg-averages the heads and ships the result back. The backbone
never moves.

## Papers

- [F170 v0.1](paper-479.md) — initial architecture + federated loop
- [F170 v0.2](paper-480.md) — INT4 quantization + real aggregator + 4-substrate byte-exactness

## Code

- [federated-tinyml-vessel](https://github.com/SuperInstance/federated-tinyml-vessel) — Python + C implementation
- [federated-tinyml-npm](https://github.com/SuperInstance/federated-tinyml-npm) — JS port
- [quilt-rust](https://github.com/SuperInstance/quilt-rust) — Rust crate (11/11 tests)
