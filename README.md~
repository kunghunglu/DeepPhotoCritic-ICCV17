# ICCV17 - DeepPhotoCritic

Aesthetic Critiques Generation for Photos

Created by Kuang-Yu Chang\*, Kung-Hung Lu\* and Chu-Song Chen at Academia Sinica, Taipei, Taiwan

## Introducion

This work proposed a paragraph-to-sentence captioning model to generate aesthetic-oriented captions for images.
There are various ways to comment an image(e.g. lighting, composition and subject...), especially in aesthetic quality analysis. Owing to the nature of multi-aspect for photo critique, we assume there is an input dataset of images with comments on various aspects. Our approach, aspect fusion(AF) could fuse serveral comments from different aspects for an image and exploit an attention mechenism to generate appropriate photo critique. Moreover, AF could produce more diverse captions than simple CNN-LSTM model and thus is favorable to human. All the training is of end-to-end manner. The implementation is used with Torch framework and based on [neuraltalk2](http://github.com/karpathy/neuraltalk2.

## Citation

If you find Ordered Weighted Averagin Layer useful in your research, please consider citing:

	Aesthetic Critiques Generation for Photos
	Kuang-Yu Chang*, Kung-Hung Lu*, and Chu-Song Chen 	
	IEEE International Conference on Computer Vision(ICCV), 2017


## Installation

### Prerequisition
Follow the Neuraltalk2 [requirements](http://github.com/karpathy/neuraltalk2) to set up Torch

### Training

run the script `train.sh`. Use `aspect_net` to load the pretrained model for the attribute nets and `dec_net` for the fusion net. 

	$ ./train.sh

### Evaluation

run the script `eval.sh`. You can set the testing image folder and the number of images you want to eval. If you want to visualize your images in order, you can use an extra `index_json` file (refer to the sample file  `vis/vis_label_list.json`) to load images in that order. Finally, it will create an `vis.json` inside the `vis` folder for visualization.

	$ ./eval.sh

## Resources in this paper

comming soon...

## Contact 

Please feel free to leave suggestions or comments to Kung-Hung Lu (henrylu@iis.sinica.edu.tw), Kuang-Yu Chang (kuangyu@iis.sinica.edu.tw) and Chu-Song Chen (song@iis.sinica.edu.tw)

