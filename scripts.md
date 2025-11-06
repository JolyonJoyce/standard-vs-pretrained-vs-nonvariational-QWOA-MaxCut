# Pretrained parameters:
```bash
sbatch gpu-runQAOAmc-auxillary-graphs.slurm "cycle" 16 ; sbatch gpu-runQAOAmc-auxillary-graphs.slurm "complete" 16 ; sbatch gpu-runQAOAmc-auxillary-graphs.slurm "path" 18 ; sbatch gpu-runQAOAmc-auxillary-graphs.slurm "cycle" 18 ; sbatch gpu-runQAOAmc-auxillary-graphs.slurm "complete" 18
```

---

# Results:

## stdQAOAmc and nvQWOAmc for n=18:
Test:
```bash
sbatch gpu-runQAOAmc.slurm "nv" 18 0
```

Final:
```bash
for i in {0..199} ; do sbatch gpu-runQAOAmc.slurm "std" 18 $i ; sbatch gpu-runQAOAmc.slurm "nv" 18 $i ; done
```

## ptQAOAmc for n=16:
```bash
for i in {0..199} ; do sbatch gpu-runQAOAmc.slurm pt 16 $i cycle ; sbatch gpu-runQAOAmc.slurm pt 16 $i complete ; sbatch gpu-runQAOAmc.slurm pt 18 $i path ; sbatch gpu-runQAOAmc.slurm pt 18 $i cycle ; sbatch gpu-runQAOAmc.slurm pt 18 $i complete ; done
```
