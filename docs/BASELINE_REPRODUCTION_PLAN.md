# Published baseline reproduction status

The implemented feedback/guarded policies are local diagnostic controls. They
must not be relabeled AdaScale, ApproxDet, or Chanakya.

Verified source snapshots downloaded November 13, 2026:

- [Chanakya upstream](https://github.com/microsoft/Chanakya), commit
  `fbb121697f944a5ce5a4edeceba5ce88ae76fc09`, local source at
  `/media/imsel/4TBSSD/EMCH585_Artifacts/baselines/Chanakya`.
  Its supplied environment pins Python 3.7 and older dependencies; inspect and
  isolate its environment rather than replacing the working Jetson stack.
- [ApproxDet upstream](https://github.com/purdue-dcsl/ApproxDet), commit
  `d3c745a8d70a35dddcab0d6f4173a51c682d043a`, local source at
  `/media/imsel/4TBSSD/EMCH585_Artifacts/baselines/ApproxDet`.
  Its published setup targets TX2, TensorFlow 1.14 and old NumPy. It explicitly
  requires device-specific profiling/modeling for another platform.
- [AdaScale paper](https://proceedings.mlsys.org/paper_files/paper/2019/hash/b3ac5aacb06c91fda1af776a677100ab-Abstract.html)
  remains a candidate content-adaptive comparison. Do not substitute a similarly
  named OOD/classification repository for the video-detection method.

Downloaded code is preparation, not a successful reproduction. No upstream
training or inference has been run. Workstation authentication and three-family
GPU training/evaluation are complete. Original upstream model/data assets,
legacy runtime isolation and comparable multimodal execution remain open.

November 23 progress: the upstream ApproxDet model bundle was downloaded and
its seven regular model assets extracted into a separate artifact directory.
The archive SHA-256 and member inventory are in
`results/processed/baseline_preparation_20261123/status.json`. This is asset
preparation, not a successful baseline execution. The upstream contention
cleanup uses pattern-wide `pkill`; replace it with cleanup restricted to owned
processes before execution on shared hardware. Original ImageNet-VID case data,
compatible runtime validation and reprofiled multimodal integration remain open.

Required execution steps:

1. Inspect each upstream license, model/data assets and environment. Reproduce
   an original provided case before porting it to the multimodal detector.
2. Keep an isolated, pinned workstation environment for older dependencies.
   Verify compatibility on the actual available GPU/driver rather than assuming
   the original TX2 CUDA/TensorFlow instructions work on RTX/Ada hardware.
3. Document any restricted action space or detector/interface replacement.
   Distinguish a policy adaptation from reproduction of the complete system.
4. Fit performance models and learned policies on train/validation data only.
   Account for probing/profiling, tracking, policy evaluation, memory and switches.
5. Use the same registered inputs, declared class subset, splits, arrival trace,
   detector weights and deployment budget wherever a comparison permits them.
   Report incompatibilities instead of quietly granting unequal capabilities.
6. Evaluate latency/energy and causal streaming accuracy on held-out sequences.
   Include fixed frontiers and negative/tied results with run/sequence uncertainty.

Prepared additional-family training entry point:

```bash
python -m experiments.train_families --family fasterrcnn \
  --device cuda:AVAILABLE_GPU --data-root /path/to/M3FD_Detection \
  --output /new/run/path --epochs 100 --max-hours 2
python -m experiments.train_families --family retinanet \
  --device cuda:AVAILABLE_GPU --data-root /path/to/M3FD_Detection \
  --output /different/new/run/path --epochs 100 --max-hours 2
```

Replace `AVAILABLE_GPU` after inspecting workstation utilization. These commands
must run on the authorized workstation. The entry point rejects CPU training.
The recipes use the existing grouped M3FD split, preserve B,G,R,IR channel order,
reserve model label zero and map the six project classes to 1–6. Validation AP
selects `best.pt`; test images are not loaded. The time limit is checked between
batches; a final validation pass can extend elapsed time. These recipes ran on
the workstation: Faster R-CNN completed 62 full epochs plus a partial epoch 63,
and RetinaNet completed 60 full epochs. Frozen validation-best checkpoints were
evaluated on M3FD and KAIST; RGBT-Tiny transfer was added November 23. Training
budgets remain unmatched to the YOLO recipe. A trained checkpoint uses the common experiment adapter
`--adapter experiments.detector_families:create --weights best.pt --device cuda:N`.
