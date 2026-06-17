# LaTeX Notes

Files in this folder describe the implemented QuanMambaScrib framework in paper style.

- `quanmambascrib_method.tex`: method section intended to be included in a CVPR/MICCAI/TMI manuscript.
- `standalone_quanmambascrib_method.tex`: minimal standalone wrapper to compile the method section directly.

The text is written to match the current ACDC implementation in:

- `code/train/train_quanmambascrib_acdc.py`
- `code/networks/quan_mamba_scrib.py`
- `code/networks/qpim.py`
- `code/utils/quan_mamba_pseudo.py`
- `code/utils/quan_mamba_losses.py`

The description assumes scribble-supervised segmentation with:

- `K` semantic classes
- unlabeled pixels marked by `ignore_index = K`
- two segmentation branches: U-Net and Mamba-UNet
- QPIM-based prototype verification
- agreement-disagreement pseudo-label construction with quantum filtering
