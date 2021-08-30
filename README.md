
This repo contains several algorithms for multi-hop reasoning on knowledge graphs.

**Models**
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [NewLook](http://tonghanghang.org/pdfs/kdd21_newlook.pdf)
- [x] [LogicE](https://arxiv.org/abs/2103.00418)

**KG Data**

The KG data (FB15k, FB15k-237, NELL995) mentioned should be put into under 'data/' folder.



**Examples**

Please refer to the `examples.sh` for the scripts of all 3 models on all 3 datasets.
The detailed setting of hyper-parameters are in /config folder:

If you want to train models, the command will be:

python main.py --config config/default.yaml
python main.py --config config/Query2Box.yaml
python main.py --config config/NewLook.yaml
python main.py --config config/Logic.yaml


If you need to evaluate on the EFO-1-QA benchmark, be sure to load from existing model checkpoint, you can
train one on your own or download
from [here](https://drive.google.com/drive/folders/13S3wpcsZ9t02aOgA11Qd8lvO0JGGENZ2?usp=sharing):

python main.py --config config/benchmark_beta.yaml --checkpoint_path ckpt/FB15k/Beta_full
python main.py --config config/benchmark_NewLook.yaml --checkpoint_path ckpt/FB15k/NLK_full --load_step 450000
python main.py --config config/benchmark_Logic.yaml --checkpoint_path ckpt/FB15k/Logic_full --load_step 450000


**Citations**

If you use this repo, please cite the following paper.

