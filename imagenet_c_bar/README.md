# CIFAR-10/ImageNet-C-Bar

This code may be used to generate CIFAR-10/ImageNet-C-Bar datasets or to test on them in-memory.  The CIFAR-10 and ImageNet datasets are required to create their respective corrupted datasets.  To generate a dataset, run

```
python make_cifar10_c_bar.py --cifar_dir <CIFAR-10_PATH> --out_dir <CIFAR-10-C-BAR_PATH>
```

and similarly for `make_imagenet_c_bar.py`.  The script uses a PyTorch dataloader to parallelize image generation, and the batch size and number of dataloader workers can be changed with the `--batch_size` and `--num_workers` flags.  Note that ImageNet-C-Bar generation can be quite slow.  To generate a subset of the dataset, a different set of corruptions can be chosen using the `--corruption_file` flag. This can be used to parallelize construction if desired.

The file `test_c_bar.py` provides a function for testing a provided model on CIFAR-10/ImageNet-C-Bar in-memory.  Note that for ImageNet, in-memory and saved results may differ due to the JPEG compression associated with saving the images.

<p align="center"><img src="../figs/new_datasets.png" data-canonical-src="../figs/new_datasets.png" height="400" /></p>

\* Base example images copyright Sehee Park and Chenxu Han.
