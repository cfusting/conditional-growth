### Running on an HPC

Most clusters do not allow non-root users to use privileged tools like Docker. [Singularity](https://sylabs.io/singularity/), however, can be used in a similar if slightly less convenient manner. Because some actions that singularity must take require root privileges, a pre-built image is provided. The image is updated automatically when the main branch of this project changes, but feel free to raise an issue if it seems out of date.

#### Prepare

Get on to a node on your cluster where you can download and test the image. You'll probably need to load a recent version of Singularity:

```bash
module load Singularity/3.6.1
```

#### Pull

Pull the image to the cluster:

```bash
singularity pull shub://cfusting/conditional-growth
```

#### Run

You can now run the above example similarly:

```bash
singularity exec --writable-tmpfs shub://cfusting/conditional-growth python /root/conditional-growth/experiments/grow/optimize_grid.py
```

#### Caveats

Singularity containers are read-only which can cause issues if you wish to make changes to (for example) the experiments configurations in optimize_grid.py. One simple solution is to enter the container interactively, copy the conditional-growth repository into your home folder, (which the container can see) and run experiments from there:

```bash
cd ~
singularity shell --writable-tmpfs shub://cfusting/conditional-growth
cp -R /root/conditional-growth .
exit
singularity exec --writable-tmpfs shub://cfusting/conditional-growth python ~/conditional-growth/experiments/grow/optimize_grid.py
```

You could of course remain within the shell and run from there.
