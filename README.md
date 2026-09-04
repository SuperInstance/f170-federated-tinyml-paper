# F170 — Federated TinyML for the Vessel Edge

The on-device learning loop for the vessel-agent system.

Every device on a fishing boat runs the same frozen audio backbone
and learns a tiny 325-parameter classifier head on its own audio.
Once a round, the device ships its 1.3 KB head to the aggregator.
The aggregator FedAvg-averages the heads and ships the result back.
The backbone never moves.

## See also

- [F170 paper](paper-479.md) — the full research writeup
- [federated-tinyml-vessel](https://github.com/SuperInstance/federated-tinyml-vessel) — the working code
- [Live Canon F170](https://live-canon.superinstance.dev/api/canon/claim?topic=federated+tinyml) — the live entry
