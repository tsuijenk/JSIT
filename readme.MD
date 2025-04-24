
## `readme_compiling_JSIT.MD`

### Environment

These instructions are tailored for the environment on the **"numbers" cluster** using:
- MATLAB R2016a installed at `/gsc/software/linux-x86_64-centos6/matlab-r2016a/`
- No need for a license (i.e. activation code) **after compiling** — thanks to the MATLAB Runtime (MCR)
- Cluster jobs should call `run_JSIT.sh`, not the binary directly

---

### Step 1: Compile `JSIT.m` using MATLAB Compiler

From the directory `/path/to/JSIT` where `JSIT.m` is located, run:

```bash
/gsc/software/linux-x86_64-centos6/matlab-r2016a/bin/mcc -m JSIT.m
```

This will generate the following files:

- `JSIT` — the compiled executable
- `JSIT.ctf` — compressed code archive (required at runtime)
- `run_JSIT.sh` — helper script to run with MATLAB Runtime
- Other supporting files like `readme.txt` and `requiredMCRProducts.txt`

>  Make sure `JSIT.ctf` is **in the same directory** as `JSIT` and `run_JSIT.sh`.

---

### Step 2: Set environment variables before running

The compiled binary requires the **MATLAB Runtime (MCR)** to be available.

Use this setup if calling `JSIT` manually:
```bash
export MCRROOT=/gsc/software/linux-x86_64-centos6/matlab-r2016a
export LD_LIBRARY_PATH=$MCRROOT/v901/runtime/glnxa64:\
$MCRROOT/v901/bin/glnxa64:\
$MCRROOT/v901/sys/os/glnxa64:\
$MCRROOT/v901/sys/opengl/lib/glnxa64:$LD_LIBRARY_PATH
```

> These are set automatically when using `run_JSIT.sh` — no need to export manually in SLURM.

---

### Step 3: Run JSIT via the helper script

Run the program using:

```bash
/path/to/JSIT/run_JSIT.sh /gsc/software/linux-x86_64-centos6/matlab-r2016a \
  /path/to/aligned_images_{fov:03}_{z:02}.tiff.mat \
  /path/to/codebook.mat \
  /path/to/output_folder_{fov:03}_{z:02}
```

- Argument 1: path to MCR (always `/gsc/software/linux-x86_64-centos6/matlab-r2016a`)
- Argument 2: `.tiff.mat` input file
- Argument 3: codebook `.mat` file
- Argument 4: output folder (must exist or be creatable)

---

### Notes

- Do **not** run `./JSIT` directly — use `run_JSIT.sh`
- You **do not need a MATLAB license** to run after compilation
- Make sure all input/output paths are accessible from compute nodes
- If you see a `CTF not found` error, make sure `JSIT.ctf` is in the same directory as `JSIT`
