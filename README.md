hello! this is a personal project on exploring rl infra tooling. almost entirely handwritten with some exceptions.
the goal is to create a a codebase I can play with to implement papers & new ideas (intermittently, bottlenecked by GPU access)
<img width="2072" height="1406" alt="dominion_arch" src="https://github.com/user-attachments/assets/173ac24a-aea9-41bb-ade8-edc1c8e1c767" />

as of now:
- workers: stateless, take env-steps. no policy copy, can scale to n workers
- inference server: centralized policy forward pass that receives weights from learner, sends actions to workers
- learner: receive trajectory batches, runs updates
- supports training in sim

planned additions:
- batched inference
- configurable colocation or decoupling
- improved importance sampling correction
- better overall versioning and tagging hygeine (datasets, training runs, etc)
- structured data buffer over generic queue
- improved weight sync (broadcasting and shared memory)
- quantized inference
- QAT support
- on policy distillation
- evaluation in sim
