Assignment 3
===================================




##  1. Differentiable Volume Rendering
You can run the code for part 1 with:

```bash
python main.py --config-name=box
```
The code renders a spiral sequence of the volume in `images/part_1.gif`.

##  2. Optimizing a basic implicit volume


Once you've done this, you can run train a model with

```bash
python main.py --config-name=train_box
```

The code renders a spiral sequence of the optimized volume in `images/part_2.gif`.


##  3. Optimizing a Neural Radiance Field (NeRF) (30 points)

You can train a NeRF on the lego bulldozer dataset with

```bash
python main.py --config-name=nerf_lego
```

After training, a spiral rendering will be written to `images/part_3.gif`.

##  Note

1. Sometimes the NeRF output can be blank.If that happens, killing and rerunning the program will give a output.
